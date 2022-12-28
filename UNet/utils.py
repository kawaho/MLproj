import os as os
import numpy as np
import glob
import re
import torch.distributed as dist
import torch.nn as nn
import torchvision
import torch
import torchvision.datasets as datasets
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim 
import torchvision.transforms as transforms
from collections import OrderedDict
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image

image_size = 96
log_interval = 10
learning_rate = 0.1
momentum = 0.5

class DataLoaderSegmentation(data.Dataset):
    def __init__(self, folder_path):
        super().__init__()
        self.img_files = glob.glob(os.path.join(folder_path,'images','*.jpg'))
        self.mask_files = []
        for img_path in self.img_files:
            self.mask_files.append(os.path.join(folder_path,'annotations/trimaps',os.path.basename(img_path.replace('jpg','png'))))

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
           
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path)

            data = transforms.Compose([transforms.Resize((image_size,image_size)),transforms.ToTensor()])(image)
            label = transforms.Compose([transforms.Resize((image_size,image_size)),transforms.PILToTensor()])(mask) - 1

#            data = torchvision.io.read_image(img_path)
#            label = torchvision.io.read_image(mask_path)-1
            return data, label

    def __len__(self):
        return len(self.img_files)

    def getpath(self, index):
      return self.img_files[index]

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29505'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def prepare(rank, world_size, dataset, batch_size=16, pin_memory=True, num_workers=2, drop_last=False):
    #Load training/testing dataset into tensors
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, drop_last=drop_last)
    
    dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, shuffle=False, sampler=sampler, drop_last=drop_last) 
    return dl

def onehot(target):
    target_onehot = torch.nn.functional.one_hot(target.type(torch.int64))
    target_onehot = torch.squeeze(target_onehot, dim=1).permute(0, 3, 1, 2)
    return target_onehot

def KL(alp_til, rank):
    ones = torch.Tensor(np.ones([1]+list(alp_til.shape)[1:])).to(rank)

    S_alpha = torch.sum(alp_til, 1, keepdim=True)
    S_ones = torch.sum(ones, 1, keepdim=True)

    lnD = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alp_til), 1, keepdim=True)
    lnD_uni = torch.sum(torch.lgamma(ones), 1, keepdim=True) - torch.lgamma(S_ones)
    
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alp_til)
    
    kl = torch.sum((alp_til - ones)*(dg1 - dg0), 1, keepdim=True) + lnD + lnD_uni
    return kl

def mseKL_loss(alpha, truth, epoch, rank, mean=True): 
    S = torch.sum(alpha, 1, keepdim=True) 
    E = alpha - 1
    pred = alpha / S
    err = torch.sum((truth-pred)**2, 1, keepdim=True) 
    var = torch.sum(alpha*(S-alpha)/(S*S*(S+1)), 1, keepdim=True) 
    
    annealing_coef = min(1.0, epoch/10)
    
    alp_til = E*(1-truth) + 1 
    penalty =  annealing_coef * KL(alp_til, rank)

    if mean:
      return torch.mean(err + var + penalty)
    return torch.sum(err + var + penalty)

def IoU(pred, truth):
    offset = 1e-6
    pred_onehot, truth_onehot = onehot(pred), onehot(truth)
    nom, denom = torch.logical_and(pred_onehot, truth_onehot), torch.logical_or(pred_onehot, truth_onehot)

    return torch.mean(torch.sum(nom, dim = (2,3))/ (torch.sum(denom, dim = (2,3))+offset))

def runTest(model, test_dl, rank, epoch, world_size):
    model.eval()
    average_loss = 0
    test_loss, test_IoU = 0, 0
    counter = 0
    test_gather_object = [None for _ in range(world_size)]

    with torch.no_grad():
        for data, target in test_dl:
            counter+=len(data)
            data, target = data.to(rank), target.to(rank)
            output = model(data)
            test_loss += F.cross_entropy(output, torch.squeeze(target).long(), reduction='sum')
            test_IoU += IoU(output.argmax(1, keepdim=True), target)

    loss_info = {'size':counter, 'loss': test_loss.item(), 'IoU': test_IoU.item()}

    torch.cuda.set_device(rank)
    if(rank == 0):
        dist.gather_object(loss_info, object_gather_list=test_gather_object)
        total_size, total_loss, total_IoU = 0, 0, 0
        for losses in test_gather_object:
          total_size+=losses['size']
          total_loss+=losses['loss']
          total_IoU+=losses['IoU']

        average_loss = total_loss / (image_size*image_size*total_size)
        average_IoU = total_IoU / (world_size*len(test_dl))
        print("Test average loss, average IoU: {}  {}\n".format(average_loss, average_IoU))
    else:
        dist.gather_object(loss_info)
    return average_loss

def runTrain(model, optimizer, train_dl, rank, epoch, writer, world_size):

    model.train()

    train_dl.sampler.set_epoch(epoch)       

    counter = 0
    train_gather_object = [None for _ in range(world_size)]

    for batch_idx, (data, target) in enumerate(train_dl):
#        print(batch_idx)
        data, target = data.to(rank), target.to(rank)

        optimizer.zero_grad()
        output = model(data) + 1
        target_onehot = onehot(target) 
        loss = mseKL_loss(output, target_onehot, epoch, rank)
#        counts = torch.unique(target, return_counts=True)[1]

#        loss = F.cross_entropy(output, target.squeeze_().long())#, weight = 1/counts)

        loss.backward()
        optimizer.step()

        counter += len(data)

        loss_info = {'size':len(data), 'loss': loss.item()}

        if batch_idx % log_interval == 0:
            torch.cuda.set_device(rank)
            if(rank == 0):
                dist.gather_object(loss_info, object_gather_list=train_gather_object)
                total_size, total_loss = 0, 0
                for losses in train_gather_object:
                    total_size+=losses['size']
                    total_loss+=losses['loss']*losses['size']
             
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, counter*world_size, len(train_dl.dataset),
                    100. * batch_idx / len(train_dl), total_loss / total_size))

                step = counter*world_size + len(train_dl.dataset)*(epoch-1)

                writer.add_scalar('loss/train', total_loss / total_size, step)
            else:
                dist.gather_object(loss_info)

def weights_init(m):
    classname = m.__class__.__name__
    #He initialization
    if classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_normal_(m.weight)#, mode='fan_in', nonlinearity='relu')
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight)#, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
#    if classname.find('BatchNorm') != -1:
#        nn.init.normal_(m.weight, 1.0, 0.02)
#        nn.init.constant_(m.bias, 0)

def runmain(rank, model, trainset, testset, world_size, n_epochs, modelargs):

    # setup the process groups
    setup(rank, world_size)
    
    # prepare the dataloader
    train_dl = prepare(rank, world_size, trainset, num_workers=world_size)
    test_dl = prepare(rank, world_size, testset, drop_last=True, num_workers=world_size)

    # instantiate the model(it's your own model) and move it to the right device
    model = model(**modelargs).to(rank)
    model.apply(weights_init)
 
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    model = DDP(model, device_ids=[rank], output_device=rank)#, find_unused_parameters=True)

    #optimizer = optim.SGD(model.parameters(), lr=learning_rate/2, momentum=momentum)
    optimizer = optim.Adam(model.parameters())

    #Logging with tensorboard
    from torch.utils.tensorboard import SummaryWriter
    
    layout = {
        "MNIST": {
            "loss": ["Multiline", ["loss/train", "loss/test"]],
        },
    }
    
    writer = SummaryWriter()
    writer.add_custom_scalars(layout)
#    model.load_state_dict(torch.load('model_unet.pt'), strict=False)
    for epoch in range(1, n_epochs + 1):
        runTrain(model, optimizer, train_dl, rank, epoch, writer, world_size)
        test_loss = runTest(model, test_dl, rank, epoch, world_size)
        if rank==0:
            step = len(train_dl.dataset)*epoch 
            writer.add_scalar('loss/test', test_loss, step)
    if rank==0: 
        torch.save(model.module.state_dict(), f"model_unet.pt")
#        for name, param in model.module.named_parameters():
#            if param.requires_grad:
#                print(name, param.data)
    cleanup()

if __name__ == '__main__':
    # suppose we have 2 gpus
    world_size = 2

    model = model_softm()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    mp.spawn(
        runmain,
        args=[model, optimizer, world_size, 3],
        nprocs=world_size,
        join=True
    )

