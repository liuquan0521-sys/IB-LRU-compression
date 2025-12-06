
from cmath import exp
import tqdm
import argparse
import math
import random
import shutil
import sys
sys.path.append('../')
import time
import os

import logging
from datetime import datetime
from PIL import Image
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile, Image
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from compressai.datasets import ImageFolder
from compressai.zoo import image_models
from utils.dataload import PromptTrainDataset
import yaml
from utils.Args import train_options
from utils.utils import configure_optimizers, AverageMeter, CustomDataParallel, save_checkpoint, configure_optimizers_prompt
from utils.metrics import RateDistortionLoss
from utils.logger import setup_logger
import numpy as np
from utils.training import train_one_epoch_c,train_one_epoch_GS
from utils.testing import test_epoch_c,test_epoch_GS
from model.TIC_prompt import TIC_Prompt
import time
import torch.autograd.profiler as profiler
import matplotlib.pyplot as plt
from collections import OrderedDict
# import seaborn as sns
import numpy as np
from compressai.zoo import image_models

def main():
    torch.backends.cudnn.benchmark = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None
    args = train_options()
    if args.seed is not None:
        seed = args.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  
        random.seed(seed)
        np.random.seed(int(seed))
        torch.backends.cudnn.deterministic = True
        
    print(torch.cuda.is_available())
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    
    #######  log   ###########
    base_dir = args.base_dir
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    tb_dirpath = os.path.join(base_dir, args.experiment, 'tb_logger')
    if not os.path.exists(tb_dirpath):
        os.makedirs(tb_dirpath)
    tb = SummaryWriter(tb_dirpath)
    
    if not os.path.exists(os.path.join(base_dir, args.experiment)):
        os.makedirs(os.path.join(base_dir, args.experiment))
        
    if not os.path.exists(os.path.join(base_dir, args.experiment, 'checkpoints')):
        os.makedirs(os.path.join(base_dir, args.experiment, 'checkpoints'))
    setup_logger('train', os.path.join(base_dir, args.experiment), 'train_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    setup_logger('val', os.path.join(base_dir, args.experiment), 'val_' + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    
    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')

    ####### training setting #########
    train_dataset = PromptTrainDataset(args, "train")
    test_dataset = PromptTrainDataset(args, "val")
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True,pin_memory=True,)
    val_dataloader = DataLoader(test_dataset,
                                batch_size=args.test_batch_size,
                                num_workers=args.num_workers,
                                shuffle=False,pin_memory=True,)

    net = TIC_Prompt(config=args)
    net = net.to(device)
            
    optimizer, aux_opt = configure_optimizers_prompt(net, args)
    rdcriterion = RateDistortionLoss(lmbda=args.lmbda)
    start_epoch = 0
    best_loss = 1e10
    current_step = 0
    
    if args.checkpoint_prompt is not None:
        checkpoint = torch.load(args.checkpoint_prompt, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        print(start_epoch)
        net.load_state_dict(checkpoint["state_dict"], strict=True)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80,100], gamma=0.1)
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        lr_scheduler._step_count = checkpoint['lr_scheduler']['_step_count']
        lr_scheduler.last_epoch = checkpoint['lr_scheduler']['last_epoch']
        best_loss = checkpoint['loss']
        current_step = start_epoch  * math.ceil(len(train_dataloader) )
        print(current_step)
        for k, p in net.named_parameters():
            if 'prompt' not in k :
                p.requires_grad = False
    
        
    logger_train.info(f"Seed: {seed}")
    logger_train.info(args)
    logger_train.info(net)
    logger_train.info(optimizer)
    logger_train.info(aux_opt)
    logger_train.info(current_step)
    logger_train.info(device)
    all_params = itertools.chain(net.prompt_gen.parameters(), net.prompt_proj.parameters())
    logger_train.info(sum(p.numel() for p in net.parameters()))
    logger_train.info(sum(p.numel() for p in all_params))
    optimizer.param_groups[0]['lr'] = args.learning_rate

    for epoch in range(start_epoch, args.epochs):
        
        current_step=train_one_epoch_GS(
            epoch, 
            net,
            rdcriterion,
            train_dataloader,
            optimizer,
            tb,
            logger_train,
            current_step,
            kl_beta_target=args.beta
        ) 

        loss = test_epoch_GS(epoch, val_dataloader, net, rdcriterion, tb, logger_val, kl_beta_target=args.beta)
        lr_scheduler.step()
   
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        net.update(force=True)
        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    
                },
                is_best,
                os.path.join(base_dir, args.experiment, 'checkpoints', "checkpoint_%03d.pth.tar" % (epoch))
            )
            if is_best:
                logger_val.info('best checkpoint saved.')
                
if __name__ == "__main__":
    main()
