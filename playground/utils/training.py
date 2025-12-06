
import tqdm 
import torch.nn as nn
import torch.autograd.profiler as profiler
import time
from utils.metrics import calculate_kl_loss
import torch

def train_one_epoch_GS(
    epoch, model, criterion_rd,  
    train_dataloader, optimizer, tb, 
    logger_train, current_step, kl_beta_target, 
    kl_annealing_steps=30, kl_annealing=True
):
   
    model.train()
    device = next(model.parameters()).device
    if kl_annealing and kl_annealing_steps > 0 :
        beta = min(kl_beta_target, kl_beta_target * ((epoch + 1)/ kl_annealing_steps))
    else:
        beta = kl_beta_target
    
    for i, (_,d,l,label) in enumerate(train_dataloader):
        
        d = d.to(device) 
        l = l.to(device)
        optimizer.zero_grad()
 
        out_net = model(d)
        out_criterion = criterion_rd(out_net, l)
        rdloss = out_criterion["rdloss"]
        mu, log_var = out_net["mu"], out_net["log_var"]
        kl_loss = calculate_kl_loss(mu, log_var)
        total_loss = rdloss+ kl_loss * beta 

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        current_step += 1
    
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
   
        if i % 100 == 0 and i >= 100:
            tb.add_scalar('{}'.format('[train]: totalloss'), total_loss.item(), current_step)
            tb.add_scalar('{}'.format('[train]: rdloss'), out_criterion["rdloss"].item(), current_step)
            tb.add_scalar('{}'.format('[train]: bpp_loss'), out_criterion["bpp_loss"].item(), current_step)
            tb.add_scalar('{}'.format('[train]: mse_loss'), out_criterion["mse_loss"].item(), current_step)
            tb.add_scalar('{}'.format('[train]: KL_loss'), kl_loss.item(), current_step)
            logger_train.info(
                f"Train epoch {epoch}: ["
                f"{i*len(d):5d}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)] "
                f'Loss: {total_loss.item():.4f} | '
                f'rdLoss: {out_criterion["rdloss"].item():.4f} | '
                f'MSE loss: {out_criterion["mse_loss"].item():.4f} | '
                f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                f'kl loss: {kl_loss.item():.4f} | '
                f'psnr: {out_criterion["psnr"].item():.6f} | '
                f"lr: {lr:.6f} | "
                f"beta: {beta:.8f} | "
            )
    return current_step



def train_one_epoch_c(
    epoch, model, criterion_rd,  
    train_dataloader, optimizer, aux_opt, 
    tb, logger_train, current_step, 
):
    model.train()
    device = next(model.parameters()).device

    for i, sample in enumerate(train_dataloader):
        _, img = sample
        img = img.to(device)
        optimizer.zero_grad()
        aux_opt.zero_grad()
   
        out_net = model(img)
        out_criterion = criterion_rd(out_net, img)
        
        total_loss = out_criterion["rdloss"]
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        aux_loss = model.aux_loss()
        aux_loss.backward()
        optimizer.step()
        aux_opt.step()
        
        current_step += 1
           
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            
        if i % 100 == 0:
            tb.add_scalar('{}'.format('[train]: loss'), out_criterion["rdloss"].item(), current_step)
            tb.add_scalar('{}'.format('[train]: bpp_loss'), out_criterion["bpp_loss"].item(), current_step)
            tb.add_scalar('{}'.format('[train]: mse_loss'), out_criterion["mse_loss"].item(), current_step)
            logger_train.info(
                f"Train epoch {epoch}: ["
                f"{i*len(img):5d}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)] "
                f'Loss: {out_criterion["rdloss"].item():.4f} | '
                f'MSE loss: {out_criterion["mse_loss"].item():.4f} | '
                f'Bpp loss: {out_criterion["bpp_loss"].item():.2f} | '
                f"Aux loss: {aux_loss.item():.2f} | "
                f"lr: {lr:.6f} | "
            )
    return current_step