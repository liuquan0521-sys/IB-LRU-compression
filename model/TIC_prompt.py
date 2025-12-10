import torch
import torch.nn as nn
from model.layer import RSTB_Prompt, RSTB, TransformerBlock
from model.utils import conv,deconv
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from timm.models.layers import trunc_normal_
from compressai.models.utils import update_registered_buffers
import torch.nn.functional as F
import math
import time
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class SpatialAdaptiveModulation(nn.Module):
    def __init__(self, 
                 feature_channels: int, 
                 prompt_dim: int,
                 mlp_prompt_hidden_dim: int = 256, 
                 spatial_channels: int = 128,
                 mod_net_hidden_channels: int = 128, 
                 attn_channels_out = None,
                 norm_layer=nn.BatchNorm2d, 
                 activation=nn.GELU):
        """
        Args:
            feature_channels (int): C in F_insert. Also output C for gamma/beta.
            prompt_dim (int): D_p for global prompt P.
            mlp_prompt_hidden_dim (int): Hidden dim for the MLP processing P.
            mlp_prompt_layers (int): Number of hidden layers in the MLP for P.
            spatial_channels (int): Output channels of the spatial processing path.
            spatial_res_blocks (int): Number of residual blocks for spatial path.
            mod_net_hidden_channels (int): Internal hidden channels for mod_net.
            mod_net_res_blocks (int): Number of residual blocks in mod_net core.
            norm_layer: Normalization layer type 
            activation: Activation function type 
        """
        super().__init__()
        self.feature_channels = feature_channels
        self.attn_channels_out = attn_channels_out if attn_channels_out is not None else feature_channels

        
        self.mlp_mu = nn.Sequential(
            nn.Linear(prompt_dim, self.feature_channels )
        )

        self.conv_spatial = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            norm_layer(feature_channels),
            activation(),
        )
    
        combined_input_channels = feature_channels * 2
        self.mod_net = nn.Sequential(
            nn.Conv2d(combined_input_channels, mod_net_hidden_channels, kernel_size=3, padding=1),
            norm_layer(mod_net_hidden_channels),
            activation(),
            nn.Conv2d(mod_net_hidden_channels, mod_net_hidden_channels, kernel_size=3, padding=1),
            norm_layer(spatial_channels),
            activation(),
            nn.Conv2d(mod_net_hidden_channels, feature_channels*2, kernel_size=3, padding=1)
        )
        self.mlp_reliability = nn.Sequential(
            nn.Linear(prompt_dim, self.attn_channels_out),
            nn.Sigmoid()
        )

    def _initialize_weights(self):
        for m in self.mlp_reliability:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        linear_layers = [m for m in self.mlp_reliability if isinstance(m, nn.Linear)]
        if linear_layers:
            final_linear_layer = linear_layers[-1] 
            if final_linear_layer.bias is not None:
                initial_bias_value = 5.0 
                nn.init.constant_(final_linear_layer.bias, initial_bias_value)

    def forward(self, F_insert: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = F_insert.shape

        processed_P = self.mlp_mu(mu)
        processed_P_spatial = processed_P.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
        spatial_features = self.conv_spatial(F_insert)
        
        combined_input = torch.cat([spatial_features, processed_P_spatial], dim=1)
        
        gamma_beta_map_base = self.mod_net(combined_input) 
        gamma_map_base, beta_map_base = torch.chunk(gamma_beta_map_base, 2, dim=1)
        reliability_w = self.mlp_reliability(logvar)
        
        if self.attn_channels_out == 1: 
            w_expanded = reliability_w.unsqueeze(-1).unsqueeze(-1)
        elif self.attn_channels_out == self.feature_channels: 
            w_expanded = reliability_w.unsqueeze(-1).unsqueeze(-1)
        else:
            print(f"Warning: Unexpected attn_channels_out ({self.attn_channels_out}). Using global attention.")
            w_avg = torch.mean(reliability_w, dim=1, keepdim=True) 
            w_expanded = w_avg.unsqueeze(-1).unsqueeze(-1)

        gamma_map_final = 1.0 + w_expanded * (gamma_map_base - 1.0)
        beta_map_final = w_expanded * beta_map_base
        F_restored = gamma_map_final * F_insert + beta_map_final

        return F_restored, gamma_map_base, beta_map_base, w_expanded


class PromptEncoderConditionalPrior(nn.Module):
    def __init__(self, 
                 input_channels: int, 
                 prompt_dim: int, 
                 hidden_channels: int = 64, 
                 num_layers: int = 4):
        """
        Args:
            input_channels (int): Channels of input feature F.
            prompt_dim (int): Dimensionality of the Prompt P.
            hidden_channels (int): Hidden channels in the CNN body.
            num_layers (int): Number of layers in the CNN body.
        """
        super().__init__()
        self.prompt_dim = prompt_dim
        layers = []
        current_channels = input_channels
        for i in range(num_layers):
            out_channels = min(hidden_channels * (2**i), 128)
            layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_channels = out_channels
            
        self.cnn_encoder = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Output dimension of the shared part
        shared_feature_dim = current_channels

        prompt_head_hidden_dim = prompt_dim * 2 
        self.prompt_head_mlp = nn.Sequential(
            nn.Linear(shared_feature_dim, prompt_head_hidden_dim),
            nn.SiLU(),
       
        )
        self.fc_mu = nn.Linear(prompt_head_hidden_dim, prompt_dim)
        self.fc_log_var = nn.Linear(prompt_head_hidden_dim, prompt_dim)
        
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def forward(self, F: torch.Tensor) :
        """
        Args:
            F (torch.Tensor): Input feature map. Shape: [B, input_channels, H, W].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - logits (torch.Tensor): Classification logits [B, num_classes].
                - mu (torch.Tensor): Mean vector [B, prompt_dim].
                - log_var (torch.Tensor): Log variance vector [B, prompt_dim].
                - P (torch.Tensor): Sampled Prompt P [B, prompt_dim].
        """
        # Extract shared features
        h_shared_spatial = self.cnn_encoder(F)
        h_shared_pooled = self.gap(h_shared_spatial)
        h_shared_flattened = torch.flatten(h_shared_pooled, 1) # Shape: [B, shared_feature_dim]


        # IB Prompt Head 
        prompt_hidden = self.prompt_head_mlp(h_shared_flattened)
        mu = self.fc_mu(prompt_hidden)             # Shape: [B, prompt_dim]
        log_var = self.fc_log_var(prompt_hidden)     # Shape: [B, prompt_dim]

        # Sample P
        P = self.reparameterize(mu, log_var) # Shape: [B, prompt_dim]
        
        return  mu, log_var, P

class TIC_Prompt(nn.Module):
    """
    Modified from TIC (Lu et al., "Transformer-based Image Compression," DCC2022.)
    """
    def __init__(self, 
                 N = 128, 
                 M = 192, 
                 config = None, 
                 input_resolution = (256,256),
                 use_checkpoint = False,
                 prompt_dim = 128,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        
 

        ###transformer block dim
        trans_channel = [N, N, N, M, N, N]
        depths = [2, 4, 6, 2, 2, 2]
        num_heads = [8, 8, 8, 16, 16, 16]
        window_size = 8
        mlp_ratio = 2.
        qkv_bias = True
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm
        self.args = config
        self.input_resolution = input_resolution
        self.prompt_dim = prompt_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        
        #### only BLOCK in self.args.ENCODER_BLOCK or self.args.DECODER_BLOCK will be add prompt
        trans_encblocks = nn.ModuleList([RSTB(
                                        dim = trans_channel[n],
                                        input_resolution = (input_resolution[0]//2**(n+1), input_resolution[1]//2**(n+1)),
                                        depth = depths[n],
                                        num_heads = num_heads[n],
                                        window_size = window_size,
                                        mlp_ratio = mlp_ratio,
                                        qkv_bias = qkv_bias, 
                                        qk_scale = qk_scale,
                                        drop= drop_rate, 
                                        attn_drop = attn_drop_rate,
                                        drop_path = dpr[sum(depths[:n]):sum(depths[:n+1])],
                                        norm_layer = norm_layer,
                                        use_checkpoint = use_checkpoint,
                                        config=config,
                                    ) for n in range(6)])
        
        trans_decblocks = nn.ModuleList([RSTB(
                                        dim = trans_channel[n] ,
                                        input_resolution = (input_resolution[0]//2**(n+1), input_resolution[1]//2**(n+1)),
                                        depth = depths[n],
                                        num_heads = num_heads[n],
                                        window_size = window_size,
                                        mlp_ratio = mlp_ratio,
                                        qkv_bias = qkv_bias, 
                                        qk_scale = qk_scale,
                                        drop= drop_rate, 
                                        attn_drop = attn_drop_rate,
                                        drop_path = dpr[sum(depths[:n]):sum(depths[:n+1])],
                                        norm_layer = norm_layer,
                                        use_checkpoint = use_checkpoint,
                                        config=config,
                                    ) for n in range(6)])
        ###  encoder ######
        #(256,256,3)->(16,16,192)  
        self.g_a0 = conv(3, N, kernel_size=5, stride=2)
        self.g_a1 = trans_encblocks[0]
        self.g_a2 = conv(N, N, kernel_size=3, stride=2)
        self.g_a3 = trans_encblocks[1]
        self.g_a4 = conv(N, N, kernel_size=3, stride=2)
        self.g_a5 = trans_encblocks[2]
        self.g_a6 = conv(N, M, kernel_size=3, stride=2)
        self.g_a7 = trans_encblocks[3]
        #(16,16,192)->(4,4,128)
        self.h_a0 = conv(M, N, kernel_size=3, stride=2)
        self.h_a1 = trans_encblocks[4]
        self.h_a2 = conv(N, N, kernel_size=3, stride=2)
        self.h_a3 = trans_encblocks[5]
        
        #### decoder ######
        self.h_s3 = trans_decblocks[5]
        self.h_s2 = deconv(N, N, kernel_size=3, stride=2)
        self.h_s1 = trans_decblocks[4]
        self.h_s0 = deconv(N, M*2, kernel_size=3, stride=2)
        self.g_s7 = trans_decblocks[3]
        self.g_s6 = deconv(M, N, kernel_size=3, stride=2)
        self.g_s5 = trans_decblocks[2]
        self.g_s4 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s3 = trans_decblocks[1]
        self.g_s2 = deconv(N, N, kernel_size=3, stride=2)
        self.g_s1 = trans_decblocks[0]
        self.g_s0 = deconv(N, 3, kernel_size=5, stride=2)
        
        self.entropy_bottleneck = EntropyBottleneck(N) 
        self.gaussian_conditional = GaussianConditional(None)
        
        if self.args.model_type == 'prompt':
            self.prompt_gen = PromptEncoderConditionalPrior(3, 128,)
            self.prompt_proj = nn.ModuleList([SpatialAdaptiveModulation(feature_channels=trans_channel[i-1], 
                                                                        prompt_dim=self.prompt_dim, )
                                                    for i in self.args.ENCODER_BLOCK])
        self.apply(self._init_weights)
        
        if self.args.model_type == 'prompt':
            self._reinitialize_prompt_modules()
                      
    def g_a(self, x, x_size=None):
        count = 0
        attns = []
        mu = []
        log_var = []
        if x_size is None:
            x_size = x.shape[2:4]
        if self.args.model_type == 'prompt':
            mu, log_var, prompt = self.prompt_gen(x)
            P = prompt if self.training else mu

        x = self.g_a0(x)
        if 1 in self.args.ENCODER_BLOCK:
            x, _, _, _= self.prompt_proj[count](x, P, log_var)
            count += 1
        x, attn = self.g_a1(x, (x_size[0]//2, x_size[1]//2))
        attns.append(attn)
        
        x = self.g_a2(x)
        if 2 in self.args.ENCODER_BLOCK:
            x, _, _, _ = self.prompt_proj[count](x, P, log_var)
            count += 1
        x, attn = self.g_a3(x, (x_size[0]//4, x_size[1]//4))
        attns.append(attn)
        
        x = self.g_a4(x)
        if 3 in self.args.ENCODER_BLOCK:
            x, _, _, _ = self.prompt_proj[count](x, P, log_var)
            count += 1
        x, attn = self.g_a5(x, (x_size[0]//8, x_size[1]//8))
        attns.append(attn)
        
        x = self.g_a6(x)
        if 4 in self.args.ENCODER_BLOCK:
            x, _, _, _ = self.prompt_proj[count](x, P, log_var)
            count += 1
        x, attn = self.g_a7(x, (x_size[0]//16, x_size[1]//16))
        attns.append(attn)
        
        return x, attns, mu, log_var
          
    def g_s(self, x, x_size=None):
        attns = []
        if x_size is None:
            x_size = x.shape[2:4]
            
        x, attn = self.g_s7(x, (x_size[0]*1, x_size[1]*1))
        attns.append(attn)
        x = self.g_s6(x)
        
        x, attn = self.g_s5(x, (x_size[0]*2, x_size[1]*2))
        attns.append(attn)
        x = self.g_s4(x)

        x, attn = self.g_s3(x, (x_size[0]*4, x_size[1]*4))
        attns.append(attn)
        x = self.g_s2(x)
     
        x, attn = self.g_s1(x, (x_size[0]*8, x_size[1]*8))
        attns.append(attn)
        x = self.g_s0(x)
     
        return x, attns
    
    def h_a(self, x, x_size=None):
        if x_size is None:
            x_size = x.shape[-2:]
        x = self.h_a0(x)
        x, _ = self.h_a1(x, (x_size[0]//2, x_size[1]//2))
        x = self.h_a2(x)
        x, _ = self.h_a3(x, (x_size[0]//4, x_size[1]//4))
        return x
    
    def h_s(self, x, x_size=None):
        if x_size is None:
            x_size = x.shape[-2:]
        x, _ = self.h_s3(x, (x_size[0], x_size[1]))
        x = self.h_s2(x)
        x, _ = self.h_s1(x, (x_size[0]*2, x_size[1]*2))
        x = self.h_s0(x)
        return x
    
    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss
    
    def _reinitialize_prompt_modules(self):
        for module in self.prompt_proj:
            if hasattr(module, '_initialize_weights'):
                module._initialize_weights()
            else:
                print("Warning: SpatialAdaptiveModulation missing _initialize_weights method!")
          
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
     
    def forward(self, x, ):
        y, attns_a, mu, log_var = self.g_a(x)
        z = self.h_a(y)
    
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)
        scales_hat, means_hat = params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
    
        x_hat, attns_s = self.g_s(y_hat, )
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "attn_a": attns_a,
            "attn_s": attns_s,
            "mu": mu,
            "log_var": log_var,
        }
        
    def update(self, scale_table=None, force=False):
        """Updates the entropy bottleneck(s) CDF values.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (bool): (default: None)  
            force (bool): overwrite previous values (default: False)

        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.

        """
        if scale_table is None:
            scale_table = get_scale_table()
        self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated
    
    
    def load_state_dict(self, state_dict, strict=True):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict=strict)
        
    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a0.weight"].size(0)
        M = state_dict["g_a6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x, ):
        y, attns, mu, log_var  = self.g_a(x, )
        z = self.h_a(y, )

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat,)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat,)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        
        x_hat, attns_s = self.g_s(y_hat,)
        x_hat = x_hat.clamp_(0, 1)
        return {"x_hat": x_hat}