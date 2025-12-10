from utils.utils import  AverageMeter
import torch
import tqdm
from torch.amp import autocast
from pytorch_msssim import ms_ssim
from torchvision import transforms
import torch.nn as nn
import os
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
from utils.metrics import calculate_kl_loss

def torch2img(x: torch.Tensor):
    return ToPILImage()(x.clamp_(0, 1).squeeze())

def test_epoch_GS(epoch, test_dataloader, model, criterion_rd, tb, logger_val, kl_beta_target, num_images_to_log=32,):

    model.eval()
    device = next(model.parameters()).device

    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    kl_loss = AverageMeter()
    psnr = AverageMeter()
    Ms_ssim = AverageMeter()
    totalloss = AverageMeter()
    triloss = AverageMeter()
   
    images_logged = 0
    logged_images = []
    logged_true_labels = []
    logged_pred_labels = []
    with torch.no_grad():
        for i, (names, degrads, cleans,label) in enumerate(test_dataloader):
            d = degrads.to(device)
            l = cleans.to(device)
            label = label.to(device)
            with autocast(device_type='cuda'):
                out_net = model(d)
                out_criterion = criterion_rd(out_net, l)
                mu, log_val = out_net["mu"], out_net["log_var"]
                klloss = calculate_kl_loss(mu, log_val)
                m = 1 - ms_ssim(out_net["x_hat"], l, data_range=1.0)
                total_loss = kl_beta_target*klloss + out_criterion["rdloss"] 
    
            kl_loss.update(klloss.item())
            bpp_loss.update(out_criterion["bpp_loss"].item())
            mse_loss.update(out_criterion["mse_loss"].item())
            psnr.update(out_criterion['psnr'].item())
            totalloss.update(total_loss.item())
            Ms_ssim.update(m)

    tb.add_scalar('{}'.format('[val]: loss'), totalloss.avg, epoch)
    tb.add_scalar('{}'.format('[val]: bpp_loss'), bpp_loss.avg, epoch)
    tb.add_scalar('{}'.format('[val]: mse_loss'), mse_loss.avg, epoch)
    tb.add_scalar('{}'.format('[val]: psnr'), psnr.avg, epoch)
    tb.add_scalar('{}'.format('[val]: ssim'), Ms_ssim.avg, epoch)
    tb.add_scalar('{}'.format('[val]: klloss'), kl_loss.avg, epoch)
    logger_val.info(
            f"Test epoch {epoch}: Average losses: "
            f"Loss: {totalloss.avg:.4f} | "
            f"MSE loss: {mse_loss.avg:.4f} | "
            f"Bpp loss: {bpp_loss.avg:.4f} | "
            f"kl loss: {kl_loss.avg:.2f} | "
            f"PSNR: {psnr.avg:.6f} | "
            f"MS-SSIM: {Ms_ssim.avg:.6f}"
        )
    
    return totalloss.avg


def test_epoch_c(epoch, test_dataloader, model, criterion_rd, tb, logger_val, ):
    model.eval()
    device = next(model.parameters()).device

    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    Aux_loss = AverageMeter()
    psnr = AverageMeter()
    Ms_ssim = AverageMeter()
    totalloss = AverageMeter()
   
    with torch.no_grad():
        for i, sample in enumerate(test_dataloader):
            _, img = sample
            img = img.to(device)
            with autocast(device_type='cuda'):
                out_net = model(img)
                out_criterion = criterion_rd(out_net, img)
                total_loss = out_criterion["rdloss"]
                m = 1 - ms_ssim(out_net["x_hat"], img, data_range=1.0)
          
            Aux_loss.update(model.aux_loss().item())
            bpp_loss.update(out_criterion["bpp_loss"].item())
            mse_loss.update(out_criterion["mse_loss"].item())
            psnr.update(out_criterion['psnr'].item())
            totalloss.update(total_loss.item())
            Ms_ssim.update(m)
            
    tb.add_scalar('{}'.format('[val]: loss'), totalloss.avg, epoch)
    tb.add_scalar('{}'.format('[val]: bpp_loss'), bpp_loss.avg, epoch)
    tb.add_scalar('{}'.format('[val]: mse_loss'), mse_loss.avg, epoch)
    tb.add_scalar('{}'.format('[val]: psnr'), psnr.avg, epoch)
    tb.add_scalar('{}'.format('[val]: ssim'), Ms_ssim.avg, epoch)
    logger_val.info(
            f"Test epoch {epoch}: Average losses: "
            f"Loss: {totalloss.avg:.4f} | "
            f"MSE loss: {mse_loss.avg:.4f} | "
            f"Bpp loss: {bpp_loss.avg:.4f} | "
            f"Aux loss: {Aux_loss.avg:.2f} | "
            f"PSNR: {psnr.avg:.6f} | "
            f"MS-SSIM: {Ms_ssim.avg:.6f}"
        )
    
    return totalloss.avg
