import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

from mmcv.cnn import bias_init_with_prob
from mmdet.models.builder import build_loss
from mmdet.models.utils import build_transformer
from mmdet.core import multi_apply
import torchvision.models as models
from collections import OrderedDict
from typing import List, Optional, Callable
from collections import namedtuple

from mmcv.utils import Config
from models.sparse_ins import SparseInsDecoder
from .utils import inverse_sigmoid
from .transformer_bricks import *
from .embedding import TimeStepEmbedding, PoseEmbedding

_dataset = 'OpenLane'
ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


class LiquidNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LiquidNeuralNetwork, self).__init__()
        self.hidden_dim = hidden_dim

        # 输入层与隐藏层之间的可微分方程权重
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        # 隐藏层与输出层的权重
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        # 初始化隐藏状态：在每个 forward pass 中重新初始化
        hidden_state = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        outputs = []
        for t in range(seq_len):
            current_input = x[:, t, :]

            # 使用非线性微分方程更新隐藏状态 (非 in-place 操作)
            delta_state = torch.tanh(self.input_to_hidden(current_input) + hidden_state)
            hidden_state = hidden_state + delta_state

            # 计算输出
            output = self.hidden_to_output(hidden_state)
            outputs.append(output)

        # 将输出叠加，形状为 [batch_size, seq_len, output_dim]
        outputs = torch.stack(outputs, dim=1)

        # 清除隐藏状态以避免在后续批次中复用
        hidden_state = None

        return outputs


# 多尺度特征融合
class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, hidden_dim=256):
        super(MultiScaleFeatureFusion, self).__init__()
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        f1 = F.interpolate(features, scale_factor=0.5, mode='bilinear', align_corners=False)
        f1 = self.conv1(f1)

        f2 = self.conv2(features)

        f3 = F.interpolate(features, scale_factor=2, mode='bilinear', align_corners=False)
        f3 = self.conv3(f3)

        f1 = F.interpolate(f1, size=f2.shape[2:], mode='bilinear', align_corners=False)
        f3 = F.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)

        data_feature = f1 + f2 + f3

        return data_feature


class MultiDepthSamplingAlignment(nn.Module):
    
    def __init__(self, in_channels=256, embed_dim=64, depth_bins=[5, 10, 20, 35]):
        super(MultiDepthSamplingAlignment, self).__init__()
        self.depth_bins = torch.tensor(depth_bins, dtype=torch.float32)
        self.num_depth_bins = len(depth_bins)

        # Learnable depth queries [D_bin, D]
        self.depth_queries = nn.Parameter(torch.randn(self.num_depth_bins, embed_dim))

        # Linear projection for queries
        self.query_proj = nn.Linear(embed_dim, embed_dim)

        # Conv projection for Key & Value
        self.kv_proj = nn.Conv2d(in_channels, 2 * embed_dim, kernel_size=1)

        # Output projection to recover channel dimension
        self.fusion_proj = nn.Conv2d(embed_dim * self.num_depth_bins, in_channels, kernel_size=1)

    def forward(self, feat):
        """
        feat: [B, C, H, W], e.g., ResNet-50 backbone feature map (e.g., 256 channels)
        """
        B, C, H, W = feat.shape

        # Project into Key and Value
        kv = self.kv_proj(feat)         # [B, 2*D, H, W]
        
        k, v = torch.chunk(kv, 2, dim=1) # [B, D, H, W] each

        # Flatten spatial dimensions
        k = k.flatten(2).transpose(1, 2)  # [B, HW, D]
        v = v.flatten(2).transpose(1, 2)  # [B, HW, D]

        # Prepare queries from depth anchors
        q = self.query_proj(self.depth_queries)  # [D_bin, D]
        q = q.unsqueeze(0).expand(B, -1, -1)  # [B, D_bin, D]
        
        # Multi-head scaled dot-product attention (single-head here)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attn = F.softmax(attn, dim=-1)

        # Apply attention to Value
        out = torch.matmul(attn, v)  # [B, D_bin, D]
        out = out.view(B, self.num_depth_bins * out.size(-1), 1, 1)

        out = out.expand(-1, -1, H, W)
        
        out = self.fusion_proj(out)  # [B, C, H, W]
        
        fusion_test = feat + out

        return feat + out  # Residual connection


# 查询优化，结合 Liquid Neural Networks
class QueryRefinementWithLNN(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8):
        super(QueryRefinementWithLNN, self).__init__()
        # 自我注意机制
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # 前馈网络部分
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # 线性层用于投影 additional_features 到所需的形状
        
        data_number = 800
        if _dataset == 'Apollo':
            data_number = 240
        
        self.feature_projection = nn.Linear(10800, data_number)

        # 集成液态神经网络模块
        self.lnn = LiquidNeuralNetwork(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)

    def forward(self, encoded_queries, additional_features=None):
        # 执行自我注意力机制
        attn_output, _ = self.self_attn(encoded_queries, encoded_queries, encoded_queries)
        attn_output = self.norm1(attn_output + encoded_queries)

        # 融合额外特征
        if additional_features is not None:
            # 将 additional_features 投影到 [8, 800, 256] 的形状
            additional_features = additional_features.permute(1, 0, 2)  # [256, 8, 10800] -> [8, 256, 10800]
                        
            additional_features = self.feature_projection(additional_features)  # [8, 256, 10800] -> [8, 256, 800]
            additional_features = additional_features.permute(0, 2, 1)  # [8, 256, 800] -> [8, 800, 256]
            combined_features = attn_output + additional_features
        else:
            combined_features = attn_output

        # 使用液态神经网络对组合后的特征进行优化
        refined_queries = self.lnn(combined_features)

        # 前馈网络进一步处理
        refined_queries = self.ffn(refined_queries)
        refined_queries = self.norm2(refined_queries + attn_output)

        return refined_queries


class HierarchicalQueryFiltering(nn.Module):
    def __init__(self, top_k=100):
        super(HierarchicalQueryFiltering, self).__init__()
        self.top_k = top_k

    def forward(self, queries, query_scores):
        device = queries.device
        query_scores = query_scores.to(device)

        batch_size, num_queries, hidden_dim = queries.shape
        actual_top_k = min(self.top_k, num_queries)

        top_k_scores, top_k_indices = query_scores.topk(actual_top_k, dim=1, largest=True, sorted=False)

        filtered_queries = torch.gather(queries, 1, top_k_indices.unsqueeze(-1).expand(batch_size, actual_top_k, hidden_dim))

        restored_queries = torch.zeros_like(queries, device=device)

        restored_queries.scatter_(1, top_k_indices.unsqueeze(-1).expand(batch_size, actual_top_k, hidden_dim), filtered_queries)

        return restored_queries


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def exists(x):
    return x is not None


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def TransformerEncoderWrapper(d_model: int = 1024, nhead: int = 4, num_encoder_layers: int = 8, dim_feedforward: int = 1024,
                              dropout: float = 0.1, norm_first: bool = True, batch_first: bool = True):
    encoder_layer = torch.nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=batch_first,
        norm_first=norm_first,
    )

    _trunk = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers)
    return _trunk


class MLP(torch.nn.Sequential):
    
    def __init__(self, in_channels: int, hidden_channels: List[int], norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU, inplace: Optional[bool] = True,
                 bias: bool = True, norm_first: bool = False, dropout: float = 0.0):
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels

        for hidden_dim in hidden_channels[:-1]:
            if norm_first and norm_layer is not None:
                layers.append(norm_layer(in_dim))

            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))

            if not norm_first and norm_layer is not None:
                layers.append(norm_layer(hidden_dim))

            layers.append(activation_layer(**params))

            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout, **params))

            in_dim = hidden_dim

        if norm_first and norm_layer is not None:
            layers.append(norm_layer(in_dim))

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        if dropout > 0:
            layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)


class Denoiser(nn.Module):
    def __init__(self, target_dim: int = 5, pivot_cam_onehot: bool = True, z_dim: int = 1411,
                 mlp_hidden_dim: bool = 128, d_model: int = 1024):
        super().__init__()
        self.pivot_cam_onehot = pivot_cam_onehot
        self.target_dim = target_dim

        self.time_embed = TimeStepEmbedding()
        self.pose_embed = PoseEmbedding(target_dim=self.target_dim)

        first_dim = (self.time_embed.out_dim + self.pose_embed.out_dim + z_dim)

        self._first = nn.Linear(first_dim, d_model)

        self._trunk = TransformerEncoderWrapper()

        self._last = MLP(d_model, [mlp_hidden_dim, 60], norm_layer=nn.LayerNorm)

    def forward(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor):
        B, N, D = x.shape
        t_emb = self.time_embed(t)
        t_emb = self.time_embed(t).unsqueeze(1).expand(-1, N, -1)
        
        x_emb = self.pose_embed(x)
        z = z.unsqueeze(1).expand(-1, N, -1)
        feed_feats = torch.cat([x_emb, t_emb, z], dim=-1)

        input_ = self._first(feed_feats)
        feats_ = self._trunk(input_)
        output = self._last(feats_)

        return output


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps=100, sampling_timesteps=None, beta_1=0.0001, beta_T=0.1, loss_type="AtLocPlus",
                 objective="pred_noise", beta_schedule="custom", p2_loss_weight_gamma=0.0, p2_loss_weight_k=1):
        super().__init__()
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.loss_type = loss_type
        self.objective = objective
        self.beta_schedule = beta_schedule
        self.p2_loss_weight_gamma = p2_loss_weight_gamma
        self.p2_loss_weight_k = p2_loss_weight_k

        self.init_diff_hyper(self.timesteps, self.sampling_timesteps, self.beta_1, self.beta_T,
                             self.loss_type, self.objective, self.beta_schedule, self.p2_loss_weight_gamma,
                             self.p2_loss_weight_k)
        
        self.denoiser = Denoiser()

    def init_diff_hyper(self, timesteps, sampling_timesteps, beta_1, beta_T, loss_type,
                        objective, beta_schedule, p2_loss_weight_gamma, p2_loss_weight_k):
        betas = torch.linspace(
            beta_1, beta_T, timesteps, dtype=torch.float64
        )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters
        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0
        # at the beginning of the diffusion chain
        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - alphas_cumprod),
        )

        # calculate p2 reweighting
        register_buffer(
            "p2_loss_weight",
            (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod))
            ** -p2_loss_weight_gamma,
        )

        # 定义损失函数
        self.loss_fn = nn.SmoothL1Loss(beta=1.0, reduction='mean')

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return (
            posterior_mean,
            posterior_variance,
            posterior_log_variance_clipped,
        )

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def model_predictions(self, x, t, z, x_self_cond=None):
        model_output = self.denoiser(x, t, z)
    
        pred_noise = self.predict_noise_from_start(x, t, model_output)
        x_start = model_output

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(
        self,
        x: torch.Tensor,  # B x N_x x dim
        t: int,
        z: torch.Tensor,
        x_self_cond=None,
        clip_denoised=False,
    ):
        preds = self.model_predictions(x, t, z)

        x_start = preds.pred_x_start

        if clip_denoised:
            raise NotImplementedError(
                "We don't clip the output because \
                    pose does not have a clear bound."
            )

        (
            model_mean,
            posterior_variance,
            posterior_log_variance,
        ) = self.q_posterior(x_start=x_start, x_t=x, t=t)

        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,  # B x N_x x dim
        t: int,
        z: torch.Tensor,
        x_self_cond=None,
        clip_denoised=False,
        cond_fn=None,
        cond_start_step=0,
    ):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long
        )
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=batched_times,
            z=z,
            x_self_cond=x_self_cond,
            clip_denoised=clip_denoised,
        )

        if cond_fn is not None and t < cond_start_step:
            model_mean = cond_fn(model_mean, t)
            noise = 0.0
        else:
            noise = torch.randn_like(x) if t > 0 else 0.0  # no noise if t == 0

        pred = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        z: torch.Tensor,
        cond_fn=None,
        cond_start_step=0,
    ):
        batch, device = shape[0], self.betas.device

        # Init here
        pose = torch.randn(shape, device=device)

        x_start = None

        pose_process = []
        pose_process.append(pose.unsqueeze(0))

        for t in reversed(range(0, self.num_timesteps)):
            pose, _ = self.p_sample(
                x=pose,
                t=t,
                z=z,
                cond_fn=cond_fn,
                cond_start_step=cond_start_step,
            )
            pose_process.append(pose.unsqueeze(0))

        return pose, torch.cat(pose_process)

    @torch.no_grad()
    def sample(self, shape, z, cond_fn=None, cond_start_step=0):
        # TODO: add more variants
        sample_fn = self.p_sample_loop
        return sample_fn(
            shape, z=z, cond_fn=cond_fn, cond_start_step=cond_start_step
        )

    @torch.no_grad()
    def ddim_sample(self, shape, z, sampling_timesteps=20):
        batch_size = shape[0]
        device = self.betas.device
        img = torch.randn(shape, device=device)

        times = torch.linspace(-1, self.num_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in time_pairs:
            time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
            pred = self.model_predictions(img, time_cond, z)
            pred_noise = pred.pred_noise
            x_start = pred.pred_x_start

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            eta = 0

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            x_start = (img - torch.sqrt((1. - alpha)) * pred_noise) / torch.sqrt(alpha)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        return x_start, img

    def p_losses(self, x_start, t, z=None, noise=None, mask=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_out = self.denoiser(x, t, z)

        target = x_start
        x_0_pred = model_out
        loss = self.loss_fn(model_out, target)

        if mask is not None:
            loss = loss.mean(dim=-1)  # [B, Q]
            loss = (loss * mask.float()).sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss, model_out

    def forward(self, pose, z=None, mask=None, *args, **kwargs):
        b = len(pose)
        t = torch.randint(0, self.num_timesteps, (b,), device=pose.device).long()
        return self.p_losses(pose, t, z=z, mask=mask, *args, **kwargs)


class LATRHead(nn.Module):
    def __init__(self, args,
                 dim=128,
                 num_group=1,
                 num_convs=4,
                 in_channels=128,
                 kernel_dim=128,
                 positional_encoding=dict(
                    type='SinePositionalEncoding',
                    num_feats=128 // 2, normalize=True),
                 num_classes=21,
                 num_query=30,
                 embed_dims=128,
                 transformer=None,
                 num_reg_fcs=2,
                 depth_num=50,
                 depth_start=3,
                 top_view_region=None,
                 position_range=[-50, 3, -10, 50, 103, 10.],
                 pred_dim=10,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
                 loss_reg=dict(type='L1Loss', loss_weight=2.0),
                 loss_vis=dict(type='BCEWithLogitsLoss', reduction='mean'),
                 sparse_ins_decoder=Config(
                    dict(
                        encoder=dict(
                            out_dims=64),# neck output feature channels
                        decoder=dict(
                            num_group=1,
                            output_iam=True,
                            scale_factor=1.),
                        sparse_decoder_weight=1.0,
                        )),
                 xs_loss_weight=1.0,
                 zs_loss_weight=5.0,
                 vis_loss_weight=1.0,
                 cls_loss_weight=20,
                 project_loss_weight=1.0,
                 trans_params=dict(
                     init_z=0, bev_h=250, bev_w=100),
                 pt_as_query=False,
                 num_pt_per_line=5,
                 num_feature_levels=1,
                 gt_project_h=20,
                 gt_project_w=30,
                 project_crit=dict(
                     type='SmoothL1Loss',
                     reduction='none'),
                 ):
        super().__init__()
        self.trans_params = dict(
            top_view_region=top_view_region,
            z_region=[position_range[2], position_range[5]])
        self.trans_params.update(trans_params)
        self.gt_project_h = gt_project_h
        self.gt_project_w = gt_project_w

        self.num_y_steps = args.num_y_steps
        self.register_buffer('anchor_y_steps', torch.from_numpy(args.anchor_y_steps).float())
        self.register_buffer('anchor_y_steps_dense', torch.from_numpy(args.anchor_y_steps_dense).float())

        project_crit['reduction'] = 'none'
        self.project_crit = getattr(nn, project_crit.pop('type'))(**project_crit)

        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.code_size = pred_dim
        self.num_query = num_query
        self.num_group = num_group
        self.num_pred = transformer['decoder']['num_layers']
        self.pc_range = position_range
        self.xs_loss_weight = xs_loss_weight
        self.zs_loss_weight = zs_loss_weight
        self.vis_loss_weight = vis_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.project_loss_weight = project_loss_weight

        loss_reg['reduction'] = 'none'
        self.reg_crit = build_loss(loss_reg)
        self.cls_crit = build_loss(loss_cls)
        self.bce_loss = build_nn_loss(loss_vis)
        self.sparse_ins = SparseInsDecoder(cfg=sparse_ins_decoder)

        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        self.position_range = position_range
        self.depth_start = depth_start
        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.position_encoder = nn.Sequential(
            nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.transformer = build_transformer(transformer)
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        
        top_k = 400
        if _dataset == 'Apollo':
            top_k = 120
        
        self.query_filter = HierarchicalQueryFiltering(top_k=top_k)
        self.query_refinement = QueryRefinementWithLNN(hidden_dim=self.embed_dims)

        # 新增多尺度特征融合
        # self.multi_scale_fusion = MultiScaleFeatureFusion(hidden_dim=self.embed_dims)
        self.multi_scale_fusion = MultiDepthSamplingAlignment(in_channels=256, embed_dim=64)

        # build pred layer: cls, reg, vis
        self.num_reg_fcs = num_reg_fcs
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.num_classes))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(
            nn.Linear(
                self.embed_dims,
                3 * self.code_size // num_pt_per_line))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        self.num_pt_per_line = num_pt_per_line
        self.point_embedding = nn.Embedding(
            self.num_pt_per_line, self.embed_dims)

        self.reference_points = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.ReLU(True),
            nn.Linear(self.embed_dims, 2 * self.code_size // num_pt_per_line))
        self.num_feature_levels = num_feature_levels
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.refine_loss_weight = 1.0
        self.diffusion_refiners = nn.ModuleList([
            GaussianDiffusion() for _ in range(self.num_pred)
        ])

        self._init_weights()

    def _init_weights(self):
        self.transformer.init_weights()
        xavier_init(self.reference_points, distribution='uniform', bias=0)
        if self.cls_crit.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        normal_(self.level_embeds)

    def compute_query_scores(self, query_embeds):
        # 使用 L2 范数计算每个查询的显著性得分
        query_scores = torch.norm(query_embeds, dim=-1)  # 计算每个查询的 L2 范数
        return query_scores

    def forward(self, input_dict, is_training=True):
        output_dict = {}
        img_feats = input_dict['x']

        if not isinstance(img_feats, (list, tuple)):
            img_feats = [img_feats]

        # 执行稀疏实例解码
        sparse_output = self.sparse_ins(
            img_feats[0],
            lane_idx_map=input_dict['lane_idx'],
            input_shape=input_dict['seg'].shape[-2:],
            is_training=is_training)

        B, C, H, W = img_feats[0].shape
        masks = img_feats[0].new_zeros((B, H, W))

        sin_embed = self.positional_encoding(masks)
        sin_embed = self.adapt_pos3d(sin_embed)

        query = sparse_output['inst_features']  # BxNxC
        query = query.unsqueeze(2) + self.point_embedding.weight[None, None, ...]

        query_embeds = self.query_embedding(query).flatten(1, 2)

        # 多尺度特征融合
        # torch.Size([8, 256, 90, 120])
        fused_features = self.multi_scale_fusion(img_feats[0])

        # 计算查询分数
        query_scores = self.compute_query_scores(query_embeds)

        # 查询过滤
        filtered_queries = self.query_filter(query_embeds, query_scores)

        # 查询优化，传入融合后的多尺度特征
        refined_queries = self.query_refinement(filtered_queries, additional_features=fused_features.flatten(2).transpose(0, 1))
        # refined_queries = self.query_refinement(filtered_queries)

        query_embeds = refined_queries

        query = torch.zeros_like(query_embeds)
        reference_points = self.reference_points(query_embeds)
        reference_points = reference_points.sigmoid()
        mlvl_feats = img_feats

        feat_flatten = []
        spatial_shapes = []
        mlvl_masks = []

        assert self.num_feature_levels == len(mlvl_feats)
        for lvl, feat in enumerate(mlvl_feats):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(2).permute(2, 0, 1)  # NxBxC
            feat = feat + self.level_embeds[None, lvl:lvl+1, :].to(feat.device)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
            mlvl_masks.append(torch.zeros((bs, *spatial_shape),
                                           dtype=torch.bool,
                                           device=feat.device))

        if self.transformer.with_encoder:
            mlvl_positional_encodings = []
            pos_embed2d = []
            for lvl, feat in enumerate(mlvl_feats):
                mlvl_positional_encodings.append(
                    self.positional_encoding(mlvl_masks[lvl]))
                pos_embed2d.append(
                    mlvl_positional_encodings[-1].flatten(2).permute(2, 0, 1))
            pos_embed2d = torch.cat(pos_embed2d, 0)
        else:
            mlvl_positional_encodings = None
            pos_embed2d = None

        feat_flatten = torch.cat(feat_flatten, 0)

        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=query.device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1, )),
             spatial_shapes.prod(1).cumsum(0)[:-1])
        )

        pos_embed = None
        outs_dec, project_results, outputs_classes, outputs_coords = \
            self.transformer(
                feat_flatten, None,
                query, query_embeds, pos_embed,
                reference_points=reference_points,
                reg_branches=self.reg_branches,
                cls_branches=self.cls_branches,
                img_feats=img_feats,
                lidar2img=input_dict['lidar2img'],
                pad_shape=input_dict['pad_shape'],
                sin_embed=sin_embed,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                mlvl_masks=mlvl_masks,
                mlvl_positional_encodings=mlvl_positional_encodings,
                pos_embed2d=pos_embed2d,
                image=input_dict['image'],
                **self.trans_params)

        all_cls_scores = torch.stack(outputs_classes)
        all_line_preds = torch.stack(outputs_coords)
        
        """
        # [6, 8, 40, 20, 1, 3]
        # all_line_preds = all_line_preds.permute(0, 1, 2, 5, 3, 4)
        pooled_feat = F.adaptive_avg_pool2d(img_feats[0], (1, 1)).view(B, -1)
        
        refined_preds = []
        for i in range(all_line_preds.shape[0]):
            # [8, 40, 20, 1, 3]
            refine_input = all_line_preds[i]
            
            B, Q, A, P, D = refine_input.shape
            assert D == 3
            
            # [8, 40, 60]
            coarse_preds = refine_input.view(B, Q, -1)
            
            diffusion_model = self.diffusion_refiners[i]
            
            if is_training:
                gt_lanes = input_dict['ground_lanes']
                Q_gt = gt_lanes.shape[1]
                Q_target = Q
                P_gt = self.anchor_y_steps.shape[0]

                gt_x = gt_lanes[:, :, :P_gt]
                gt_z = gt_lanes[:, :, P_gt:2*P_gt]
                gt_vis = gt_lanes[:, :, 2*P_gt:3*P_gt]

                gt_x_offset = (gt_x - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
                gt_z_offset = (gt_z - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
                gt_flat = torch.cat([gt_x_offset, gt_z_offset, gt_vis], dim=-1)

                target_gt = torch.cat([
                    gt_flat,
                    torch.zeros((B, Q_target - Q_gt, P_gt * D), device=gt_flat.device)
                ], dim=1)

                supervision_mask = torch.cat([
                    torch.ones((B, Q_gt), dtype=torch.bool, device=gt_flat.device),
                    torch.zeros((B, Q_target - Q_gt), dtype=torch.bool, device=gt_flat.device)
                ], dim=1)

                residual_target = target_gt - coarse_preds.detach()
                _, residual_preds = diffusion_model(residual_target, z=pooled_feat, mask=supervision_mask)
                refined_output = coarse_preds + residual_preds
            else:
                target_shape = coarse_preds.shape
                residual_preds, _ = diffusion_model.ddim_sample(
                    shape=target_shape, z=pooled_feat, sampling_timesteps=20
                )
                refined_output = coarse_preds + residual_preds
            
            refined_output = refined_output.view(B, Q, A, P, D)
            refined_preds.append(refined_output.unsqueeze(0))
        
        all_line_preds = torch.cat(refined_preds, dim=0)
        """

        all_line_preds[..., 0] = (all_line_preds[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        all_line_preds[..., 1] = (all_line_preds[..., 1] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

        # 恢复到原始格式
        all_line_preds = all_line_preds.view(
            len(outputs_classes), bs, self.num_query,
            self.transformer.decoder.num_anchor_per_query,
            self.transformer.decoder.num_points_per_anchor, 2 + 1)
        all_line_preds = all_line_preds.permute(0, 1, 2, 5, 3, 4)
        all_line_preds = all_line_preds.flatten(3, 5)

        output_dict.update({
            'all_cls_scores': all_cls_scores,
            'all_line_preds': all_line_preds,
        })
        output_dict.update(sparse_output)

        if is_training:
            losses = self.get_loss(output_dict, input_dict)
            project_loss = self.get_project_loss(
                project_results, input_dict,
                h=self.gt_project_h, w=self.gt_project_w)
            losses['project_loss'] = \
                self.project_loss_weight * project_loss
            # losses['diffusion_refine_loss'] = refine_loss * self.refine_loss_weight
            output_dict.update(losses)
            
        return output_dict

    def get_project_loss(self, results, input_dict, h=20, w=30):
        gt_lane = input_dict['ground_lanes_dense']
        gt_ys = self.anchor_y_steps_dense.clone()
        code_size = gt_ys.shape[0]
        gt_xs = gt_lane[..., :code_size]
        gt_zs = gt_lane[..., code_size : 2*code_size]
        gt_vis = gt_lane[..., 2*code_size:3*code_size]
        gt_ys = gt_ys[None, None, :].expand_as(gt_xs)
        gt_points = torch.stack([gt_xs, gt_ys, gt_zs], dim=-1)

        B = results[0].shape[0]
        ref_3d_home = F.pad(gt_points, (0, 1), value=1)
        coords_img = ground2img(
            ref_3d_home,
            h, w,
            input_dict['lidar2img'],
            input_dict['pad_shape'], mask=gt_vis)

        all_loss = 0.
        for projct_result in results:
            projct_result = F.interpolate(
                projct_result,
                size=(h, w),
                mode='nearest')
            gt_proj = coords_img.clone()

            mask = (gt_proj[:, -1, ...] > 0) * (projct_result[:, -1, ...] > 0)
            diff_loss = self.project_crit(
                projct_result[:, :3, ...],
                gt_proj[:, :3, ...],
            )
            diff_y_loss = diff_loss[:, 1, ...]
            diff_z_loss = diff_loss[:, 2, ...]
            diff_loss = diff_y_loss * 0.1 + diff_z_loss
            diff_loss = (diff_loss * mask).sum() / torch.clamp(mask.sum(), 1)
            all_loss = all_loss + diff_loss

        return all_loss / len(results)

    def get_loss(self, output_dict, input_dict):
        all_cls_pred = output_dict['all_cls_scores']
        all_lane_pred = output_dict['all_line_preds']
        gt_lanes = input_dict['ground_lanes']
        all_xs_loss = 0.0
        all_zs_loss = 0.0
        all_vis_loss = 0.0
        all_cls_loss = 0.0
        matched_indices = output_dict['matched_indices']
        num_layers = all_lane_pred.shape[0]

        def single_layer_loss(layer_idx):
            gcls_pred = all_cls_pred[layer_idx]
            glane_pred = all_lane_pred[layer_idx]

            glane_pred = glane_pred.view(
                glane_pred.shape[0],
                self.num_group,
                self.num_query,
                glane_pred.shape[-1])
            gcls_pred = gcls_pred.view(
                gcls_pred.shape[0],
                self.num_group,
                self.num_query,
                gcls_pred.shape[-1])

            per_xs_loss = 0.0
            per_zs_loss = 0.0
            per_vis_loss = 0.0
            per_cls_loss = 0.0
            batch_size = len(matched_indices[0])

            for b_idx in range(len(matched_indices[0])):
                for group_idx in range(self.num_group):
                    pred_idx = matched_indices[group_idx][b_idx][0]
                    gt_idx = matched_indices[group_idx][b_idx][1]

                    cls_pred = gcls_pred[:, group_idx, ...]
                    lane_pred = glane_pred[:, group_idx, ...]

                    if gt_idx.shape[0] < 1:
                        cls_target = cls_pred.new_zeros(cls_pred[b_idx].shape[0]).long()
                        cls_loss = self.cls_crit(cls_pred[b_idx], cls_target)
                        per_cls_loss = per_cls_loss + cls_loss
                        per_xs_loss = per_xs_loss + 0.0 * lane_pred[b_idx].mean()
                        continue

                    pos_lane_pred = lane_pred[b_idx][pred_idx]
                    gt_lane = gt_lanes[b_idx][gt_idx]

                    pred_xs = pos_lane_pred[:, :self.code_size]
                    pred_zs = pos_lane_pred[:, self.code_size : 2*self.code_size]
                    pred_vis = pos_lane_pred[:, 2*self.code_size:]
                    gt_xs = gt_lane[:, :self.code_size]
                    gt_zs = gt_lane[:, self.code_size : 2*self.code_size]
                    gt_vis = gt_lane[:, 2*self.code_size:3*self.code_size]

                    loc_mask = gt_vis > 0
                    xs_loss = self.reg_crit(pred_xs, gt_xs)
                    zs_loss = self.reg_crit(pred_zs, gt_zs)
                    xs_loss = (xs_loss * loc_mask).sum() / torch.clamp(loc_mask.sum(), 1)
                    zs_loss = (zs_loss * loc_mask).sum() / torch.clamp(loc_mask.sum(), 1)
                    vis_loss = self.bce_loss(pred_vis, gt_vis)

                    cls_target = cls_pred.new_zeros(cls_pred[b_idx].shape[0]).long()
                    cls_target[pred_idx] = torch.argmax(
                        gt_lane[:, 3*self.code_size:], dim=1)
                    cls_loss = self.cls_crit(cls_pred[b_idx], cls_target)

                    per_xs_loss += xs_loss
                    per_zs_loss += zs_loss
                    per_vis_loss += vis_loss
                    per_cls_loss += cls_loss

            return tuple(map(lambda x: x / batch_size / self.num_group,
                             [per_xs_loss, per_zs_loss, per_vis_loss, per_cls_loss]))

        all_xs_loss, all_zs_loss, all_vis_loss, all_cls_loss = multi_apply(
            single_layer_loss, range(all_lane_pred.shape[0]))
        all_xs_loss = sum(all_xs_loss) / num_layers
        all_zs_loss = sum(all_zs_loss) / num_layers
        all_vis_loss = sum(all_vis_loss) / num_layers
        all_cls_loss = sum(all_cls_loss) / num_layers

        return dict(
            all_xs_loss=self.xs_loss_weight * all_xs_loss,
            all_zs_loss=self.zs_loss_weight * all_zs_loss,
            all_vis_loss=self.vis_loss_weight * all_vis_loss,
            all_cls_loss=self.cls_loss_weight * all_cls_loss,
        )

    @staticmethod
    def get_reference_points(H, W, bs=1, device='cuda', dtype=torch.float):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(
                0.5, W - 0.5, W, dtype=dtype, device=device)
        )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1) 
        return ref_2d


def build_nn_loss(loss_cfg):
    crit_t = loss_cfg.pop('type')
    return getattr(nn, crit_t)(**loss_cfg)