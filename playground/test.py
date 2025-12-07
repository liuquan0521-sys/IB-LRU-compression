import sys
sys.path.append('../')
import torch
from utils.utils import  AverageMeter
import tqdm
from torch.amp import autocast
from pytorch_msssim import ms_ssim
from torchvision import transforms
import torch.nn.functional as F
import time
import os
from pathlib import Path
import struct
from torchvision.transforms import ToPILImage
import torch.nn as nn
import math
from utils.Args import train_options
from model.TIC_prompt import TIC_Prompt
from utils.logger import setup_logger
import logging
from utils.dataload import PromptTrainDataset, Compress_dataset
from torch.utils.data import DataLoader
from utils.testing import test_epoch

def torch2img(x: torch.Tensor):
    return ToPILImage()(x.clamp_(0, 1).squeeze())

def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size

def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4

def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))

def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1

def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]

def write_body(fd, shape, out_strings):
    bytes_cnt = 0
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    for s in out_strings:
        bytes_cnt += write_uints(fd, (len(s[0]),))
        bytes_cnt += write_bytes(fd, s[0])
    return bytes_cnt

def read_body(fd):
    lstrings = []
    shape = read_uints(fd, 2)
    n_strings = read_uints(fd, 1)[0]
    for _ in range(n_strings):
        s = read_bytes(fd, read_uints(fd, 1)[0])
        lstrings.append([s])

    return lstrings, shape

def compress_one_image(model, x, stream_path, H, W, img_name, pad_h, pad_w):
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        out = model.compress(x)
    torch.cuda.synchronize()
    end_time = time.time()
    shape = out["shape"]
    output = os.path.join(stream_path, img_name)
    
    with Path(output).open("wb") as f:
        write_uints(f, (H, W))
        write_body(f, shape, out["strings"])
    
    size = filesize(output)
    bpp = float(size) * 8 / ((H+pad_h) * (W+pad_w))
    enc_time = end_time - start_time
    return bpp, enc_time


def decompress_one_image(model, stream_path, img_name):
    output = os.path.join(stream_path, img_name)
    with Path(output).open("rb") as f:
        original_size = read_uints(f, 2)
        strings, shape = read_body(f)
        
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        out = model.decompress(strings, shape)
    torch.cuda.synchronize()
    end_time = time.time()
    
    dec_time = end_time - start_time
    x_hat = out["x_hat"]
    x_hat = x_hat[:, :, 0 : original_size[0], 0 : original_size[1]]
    return x_hat, dec_time


def test(epoch, test_dataloader, model, logger_test, save_dir):
    
    model.eval()
    device = next(model.parameters()).device
    bpp_loss = AverageMeter()
    psnr = AverageMeter()
    Ms_ssim = AverageMeter()
    Enc_time = AverageMeter()
    Dec_time = AverageMeter()
   
    
    with torch.no_grad():
        for i, (names, degrads, cleans, labels) in enumerate(test_dataloader):
            d = degrads.to(device)
            l = cleans.to(device)
            B, C, H, W = d.shape
            pad_h = 0
            pad_w = 0
            if H % 64 != 0:
                pad_h = 64 * (H // 64 + 1) - H
            if W % 64 != 0:
                pad_w = 64 * (W // 64 + 1) - W

            d_pad = F.pad(d, (0, pad_w, 0, pad_h), mode='reflect')
            bpp, enc_time = compress_one_image(model=model, x=d_pad, stream_path=save_dir, H=H, W=W, img_name=str(names[0]), pad_h=pad_h, pad_w=pad_w)
            x_hat, dec_time = decompress_one_image(model=model, stream_path=save_dir, img_name=str(names[0]))
        
            mse = nn.MSELoss()(x_hat, l)
            p = 10 * math.log10(1/mse)
            m = ms_ssim(x_hat, l, data_range=1.0)
            bpp_loss.update(bpp)
            psnr.update(p)
            Enc_time.update(enc_time)
            Dec_time.update(dec_time)
            Ms_ssim.update(m)
            compare = torch.cat((d[0], x_hat[0], l[0]), dim=2)
            rec = torch2img(compare)
            rec.save(os.path.join(save_dir, names[0] + '.png'))
            logger_test.info(
                f"Image[{i}] | "
                f"Bpp loss: {bpp:.4f} | "
                f"PSNR: {p:.4f} | "
                f"PSNR: {m:.4f} | "
                f"Encoding Latency: {enc_time:.4f} | "
                f"Decoding latency: {dec_time:.4f}"
            )
    

    logger_test.info(
            f"Test epoch {epoch}: Average losses: "
            f"Bpp loss: {bpp_loss.avg:.4f} | "
            f"PSNR: {psnr.avg:.6f} | "
            f"MS-SSIM: {Ms_ssim.avg:.6f}"
            f"Encoding Latency: {Enc_time.avg:.4f} | "
            f"Decoding latency: {Dec_time.avg:.4f}"
            f"runtime: {Dec_time.avg + Enc_time.avg:.4f}"
        )

def test_coco(epoch, test_dataloader, model, logger_test, save_dir):
    
    model.eval()
    device = next(model.parameters()).device
    bpp_loss = AverageMeter()
    psnr = AverageMeter()
    Ms_ssim = AverageMeter()
    Enc_time = AverageMeter()
    Dec_time = AverageMeter()
   
    
    with torch.no_grad():
        for i, (names, image, clean_image) in enumerate(test_dataloader):
        
            img = image.to(device)
            clean_img = clean_image.to(device)
            B, C, H, W = img.shape
            pad_h = 0
            pad_w = 0
            if H % 64 != 0:
                pad_h = 64 * (H // 64 + 1) - H
            if W % 64 != 0:
                pad_w = 64 * (W // 64 + 1) - W
                
            img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
            bpp, enc_time = compress_one_image(model=model, x=img_pad, stream_path=save_dir, H=H, W=W, img_name=str(names[0]), pad_h=pad_h, pad_w=pad_w)
            x_hat, dec_time = decompress_one_image(model=model, stream_path=save_dir, img_name=str(names[0]))
    
            mse = nn.MSELoss()(x_hat, clean_img)
            p = 10 * math.log10(1/mse)
            m = ms_ssim(x_hat, clean_img, data_range=1.0)
            bpp_loss.update(bpp)
            psnr.update(p)
            Enc_time.update(enc_time)
            Dec_time.update(dec_time)
            Ms_ssim.update(m)
            rec = torch2img(x_hat)
            rec.save(os.path.join(save_dir, '%03d_rec.png' % i))
            logger_test.info(
                f"Image[{i}] | "
                f"Bpp loss: {bpp:.4f} | "
                f"PSNR: {p:.4f} | "
                f"PSNR: {m:.4f} | "
                f"Encoding Latency: {enc_time:.4f} | "
                f"Decoding latency: {dec_time:.4f}"
            )
    
    logger_test.info(
            f"Test epoch {epoch}: Average losses: "
            f"Bpp loss: {bpp_loss.avg:.4f} | "
            f"PSNR: {psnr.avg:.6f} | "
            f"MS-SSIM: {Ms_ssim.avg:.6f}"
        )
    
if __name__ == '__main__':
    args = train_options()
    torch.backends.cudnn.deterministic = True
    if not os.path.exists(os.path.join('../experiment', args.experiment)):
        os.makedirs(os.path.join('../experiment', args.experiment))
    setup_logger('test', os.path.join('../experiment', args.experiment), f"{args.detype}" + args.experiment, level=logging.INFO,
                        screen=True, tofile=True)
    logger_test = logging.getLogger('test')

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    net = TIC_Prompt(config=args)
    net = net.to(device)
    checkpoint = torch.load(args.checkpoint_prompt)
    start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint["state_dict"],strict=False)
    net.update(force=True)
    test_dataset = PromptTrainDataset(args, "test")
    test_dataloader = DataLoader(test_dataset,
                                batch_size=1,
                                num_workers=args.num_workers,
                                shuffle=False,
                                pin_memory=(device == "cuda"),)
    
    save_dir = os.path.join('../experiment', args.experiment, f"{args.detype}", '%02d' % (start_epoch + 1))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test(start_epoch, test_dataloader, net, logger_test, save_dir)
    
        
    