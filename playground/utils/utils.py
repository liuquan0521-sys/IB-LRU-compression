from cmath import exp
import tqdm
import argparse
import math
import random
import shutil
import sys
import os
import time
import logging
from datetime import datetime
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile, Image
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters


    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def configure_optimizers_prompt(net, args):

    target_modules = []
    if hasattr(net, 'prompt_gen'):
        target_modules.append(net.prompt_gen)
        
    if hasattr(net, 'prompt_proj'):
        target_modules.append(net.prompt_proj)
    included_params = []
    
    for module in target_modules:
       
        if module is not None and isinstance(module, nn.Module):
            for p in module.parameters():
                
                included_params.append(p)
    if not included_params:
        print("Warning: No parameters found for the main optimizer. Check module names and exclusion logic.")
        optimizer = None
    else:
        optimizer = optim.AdamW(included_params, lr=args.learning_rate, weight_decay=1e-3)

    aux_optimizer = None 

    return optimizer, aux_optimizer

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)
        
def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        best_filename = filename.replace(filename.split('/')[-1], f"checkpoint_best_loss.pth.tar")
        torch.save(state, best_filename)
