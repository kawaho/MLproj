import os as os
import re
from collections import OrderedDict
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torchvision
import torch
import torchvision.datasets as datasets
import torch.multiprocessing as mp
import torch.nn.functional as F

from models import model_softm, model_diri, mseKL_loss
import torch.optim as optim 

log_interval = 100
learning_rate = 0.1
momentum = 0.5

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def prepare(rank, world_size, dataset, batch_size=32, pin_memory=True, num_workers=0, drop_last=False):
    #Load training/testing dataset into tensors
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, drop_last=drop_last)
    
    dl = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, shuffle=False, sampler=sampler, drop_last=drop_last) 
    return dl

def onehot(target, rank):
    target_onehot = torch.zeros(target.shape[0], 10).to(rank)
    target_onehot.scatter_(1, target.unsqueeze(1), 1.0)
    return target_onehot

def runTest(model, test_dl, rank, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dl:
            
            data, target = data.to(rank), target.to(rank)
            
            output = model(data) + 1
            target_onehot = onehot(target, rank) 
            #test_loss += F.nll_loss(output, target, reduction='sum')
            test_loss += mseKL_loss(output, target_onehot, epoch, rank, mean=False)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)

    if(rank == 0):
        print("Test average loss: {}, correct predictons: {}, total: {}, accuracy: {}% \n".format(test_loss.item() / len(test_dl.dataset), correct.item(), len(test_dl.dataset),
             100.0 * correct.item() / len(test_dl.dataset)))

    return test_loss.item(), 100.0 * correct.item() / len(test_dl.dataset) 

def runTrain(model, optimizer, train_dl, rank, epoch, writer):

    model.train()

    train_dl.sampler.set_epoch(epoch)       

    for batch_idx, (data, target) in enumerate(train_dl):

        data, target = data.to(rank), target.to(rank)

        optimizer.zero_grad()
        output = model(data) + 1

        target_onehot = onehot(target, rank) 
        loss = mseKL_loss(output, target_onehot, epoch, rank)
#        loss = F.nll_loss(output, target)
#        loss = F..mse_lossoutput, target_onehot)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)

            if(rank == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_dl.dataset),
                    100. * batch_idx / len(train_dl), loss.item()))

                step = batch_idx * len(data) + len(train_dl.dataset)*(epoch-1)

                writer.add_scalar('loss/train', loss.item(), step)

def runmain(rank, model, trainset, testset, world_size, n_epochs):

    # setup the process groups
    setup(rank, world_size)
    
    # prepare the dataloader
    train_dl = prepare(rank, world_size, trainset)
    test_dl = prepare(rank, world_size, testset, drop_last=True)

    # instantiate the model(it's your own model) and move it to the right device
    model = model().to(rank)
    
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(model, device_ids=[rank])#, output_device=rank, find_unused_parameters=True)

    #optimizer = optim.SGD(model.parameters(), lr=learning_rate/2, momentum=momentum)
    optimizer = optim.Adam(model.parameters())

    #Logging with tensorboard
    from torch.utils.tensorboard import SummaryWriter
    
    layout = {
        "MNIST": {
            "loss": ["Multiline", ["loss/train", "loss/test"]],
            "accuracy": ["Multiline", ["accuracy/test"]],
        },
    }
    
    writer = SummaryWriter()
    writer.add_custom_scalars(layout)
    model.load_state_dict(torch.load('model_diri_clip.pt'), strict=False)
    for epoch in range(1, n_epochs + 1):
        runTrain(model, optimizer, train_dl, rank, epoch, writer)
        test_loss, test_acc = runTest(model, test_dl, rank, epoch)
        if rank==0:
            step = len(train_dl.dataset)*epoch 
            writer.add_scalar('loss/test', test_loss, step)
            writer.add_scalar('accuracy/test', test_acc, step)
            

    if rank==0: 

        torch.save(model.module.state_dict(), f"model_diri_clip.pt")
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

