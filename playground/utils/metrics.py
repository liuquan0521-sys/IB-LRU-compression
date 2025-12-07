import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
import math
import torch.nn.functional as F
class FeatureHook():
    def __init__(self, module):
        module.register_forward_hook(self.attach)
    
    def attach(self, model, input, output):
        self.feature = output


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")
        self.lmbda = lmbda
    
    def psnr(self, output, target):
        
        mse = (output - target).pow(2).flatten(1).mean(1)
        max_pixel = 1.
        mse = torch.clamp(mse, min=1e-10) 
        psnr_per_sample = 10 * torch.log10(max_pixel / mse)
    
        return torch.mean(psnr_per_sample)

    
    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target) 
        out["rdloss"] = out["mse_loss"] * self.lmbda * 255**2 + out["bpp_loss"]
        out["psnr"] = self.psnr(torch.clamp(output["x_hat"],0,1), target)
        
        return out
    
def calculate_kl_loss(
    mu: torch.Tensor,                     
    log_var: torch.Tensor,                 
    ) -> torch.Tensor:
  
    var = torch.exp(log_var)
    kl_div_elements = 0.5 * ( mu.pow(2) + var - log_var - 1 )
    kl_div_per_sample = torch.sum(kl_div_elements, dim=1)
    kl_loss = torch.mean(kl_div_per_sample)

    return kl_loss


