
__all__ = ['TAMoE']

from typing import Optional
import torch
from torch import nn
from ..models.layers.pos_encoding import *
from ..models.layers.basics import *
from ..models.layers.attention import *

class TAMoE(nn.Module):
    """
    Task-Adaptive Mixture of Experts (TAMoE) model.

    Output dimension:
         [bs x target_dim x nvars] for prediction
         [bs x num_patch x n_vars x patch_len] for pretrain
    """
    def __init__(self, c_in:int, target_dim:int, patch_len:int, stride:int, num_patch:int,
                 n_layers:int=2, d_model=128, n_heads=16, shared_embedding=True, d_ff:int=256,
                 norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu",
                 res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, head_dropout=0,
                 head_type="prediction", individual=False, **kwargs):

        super().__init__()

        assert head_type in ['pretrain', 'prediction'], \
            'head type should be either pretrain or prediction'

        self.head_type = head_type
        self.n_vars = c_in
        self.num_patch = num_patch
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding

        # Patch embedding
        if not shared_embedding:
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars):
                self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        use_shared_expert = kwargs.get('use_shared_expert', False)
        use_routed_expert = kwargs.get('use_routed_expert', False)

        # Transformer encoder
        self.transformer = TransformerEncoder(d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                              dropout=dropout, pre_norm=pre_norm, activation=act,
                                              res_attention=res_attention, n_layers=n_layers, store_attn=store_attn,
                                              d_task=kwargs.get('d_task', 16),
                                              # MoE parameters
                                              use_routed_expert=use_routed_expert,
                                              num_experts=kwargs.get('num_experts', 8),
                                              moe_top_k=kwargs.get('moe_top_k', 2),
                                              moe_router_fusion_mode=kwargs.get('moe_router_fusion_mode', 'concat'),
                                              moe_router_temperature=kwargs.get('moe_router_temperature', 1.0),
                                              use_shared_expert=use_shared_expert)

        # Head
        if self.head_type == "pretrain":
            self.head = PretrainHead(d_model, patch_len, head_dropout)
        elif self.head_type == "prediction":
            self.head = PredictionHead(individual, self.n_vars, d_model, num_patch, target_dim, head_dropout)

    def forward(self, z, task: Optional[str] = None, task_labels: Optional[torch.Tensor] = None):
        """
        z: [B, num_patch, n_vars, patch_len]
        task: 'pred' | 'recon' | None
        task_labels: [B, num_patch] fine-grained task IDs for MoE routing (optional)
        """
        # Auto-set task based on head_type
        if self.head_type == "pretrain" and task is None:
            task = 'recon'
        elif self.head_type == "prediction" and task is None:
            task = 'pred'

        # Encode (input already masked by MultiTaskReconCB for recon task)
        out = self._encode(z, task=task, task_labels=task_labels)

        return self.head(out)

    def _encode(self, x, task=None, task_labels=None):
        """
        Encode input patches through patch embedding, positional encoding, and transformer.

        Args:
            x: [bs, num_patch, nvars, patch_len] input patches
            task: 'pred' or 'recon'
            task_labels: [bs, num_patch] task IDs for MoE routing

        Returns:
            z: [bs, nvars, d_model, num_patch] encoded features
        """
        bs, num_patch, n_vars, patch_len = x.shape

        # Patch embedding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars):
                z = self.W_P[i](x[:,:,i,:])
                x_out.append(z)
            x = torch.stack(x_out, dim=2)
        else:
            x = self.W_P(x)
        x = x.transpose(1,2)

        u = torch.reshape(x, (bs*n_vars, num_patch, self.d_model))
        u = self.dropout(u + self.W_pos[:num_patch])

        # Prepare task labels for transformer
        encoder_task_labels = None
        if task_labels is not None:
            _, S = task_labels.shape
            if S == num_patch:
                encoder_task_labels = task_labels.unsqueeze(1).expand(bs, n_vars, num_patch).reshape(bs * n_vars, num_patch)
            else:
                encoder_task_labels = task_labels[:, 0:1].unsqueeze(1).expand(bs, n_vars, num_patch).reshape(bs * n_vars, num_patch)

        # Transformer forward
        z = self.transformer(u, task=task, task_labels=encoder_task_labels)

        z = torch.reshape(z, (-1, n_vars, num_patch, self.d_model))
        z = z.permute(0, 1, 3, 2)

        return z

    def get_moe_aux_loss(self) -> Optional[torch.Tensor]:
        """Get MoE auxiliary loss"""
        return self.transformer.get_moe_aux_loss()

    def get_routed_l2_loss(self) -> Optional[torch.Tensor]:
        """Get routed L2 loss for differential constraint."""
        return self.transformer.get_routed_l2_loss()


class PredictionHead(nn.Module):
    """
    Prediction head: maps encoder output to forecast.
    
    Input:  [bs, nvars, d_model, num_patch]
    Output: [bs, forecast_len, nvars]
    """
    def __init__(self, individual, n_vars, d_model, num_patch, forecast_len, head_dropout=0, **kwargs):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        head_dim = d_model * num_patch

        if self.individual:
            self._flatten = nn.Flatten(start_dim=-2)
            self.heads = nn.ModuleList()
            for i in range(self.n_vars):
                self.heads.append(nn.Sequential(
                    nn.Dropout(head_dropout),
                    nn.Linear(head_dim, forecast_len)
                ))
        else:
            self._flatten = nn.Flatten(start_dim=-2)
            self.head = nn.Sequential(
                nn.Dropout(head_dropout),
                nn.Linear(head_dim, forecast_len)
            )

    def forward(self, x):                     
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x forecast_len x nvars]
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self._flatten(x[:,i,:,:])
                z = self.heads[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self._flatten(x)
            x = self.head(x)
        return x.transpose(2,1)


class PretrainHead(nn.Module):
    """
    Pretrain head: reconstructs original patches from encoder output.
    
    Input:  [bs, nvars, d_model, num_patch]
    Output: [bs, num_patch, nvars, patch_len]
    """
    def __init__(self, d_model, patch_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)

    def forward(self, x):
        """
        Args:
            x: [bs, nvars, d_model, num_patch]
        Returns:
            out: [bs, num_patch, nvars, patch_len]
        """
        x = x.transpose(2,3)
        x = self.linear(self.dropout(x))
        x = x.permute(0,2,1,3)
        return x



class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None,
                 norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False,
                 d_task: int = 16,
                 # MoE parameters
                 use_routed_expert: bool = False,
                 num_experts: int = 4,
                 moe_top_k: int = 2,
                 moe_router_fusion_mode: str = 'concat',
                 moe_router_temperature: float = 1.0,
                 # MoE mode: y = y_shared + alpha * y_routed
                 use_shared_expert: bool = False):
        super().__init__()

        self.use_shared_expert = use_shared_expert

        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model, n_heads=n_heads, d_ff=d_ff, norm=norm,
                attn_dropout=attn_dropout, dropout=dropout, activation=activation,
                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                d_task=d_task,
                use_routed_expert=use_routed_expert,
                num_experts=num_experts,
                moe_top_k=moe_top_k,
                moe_router_fusion_mode=moe_router_fusion_mode,
                moe_router_temperature=moe_router_temperature,
                use_shared_expert=use_shared_expert,
            )
            for _ in range(n_layers)
        ])
        self.res_attention = res_attention
        self.use_routed_expert = use_routed_expert

    def forward(self, src: torch.Tensor, task: Optional[str] = None,
                task_labels: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None):
        """
        Args:
            src: [B, L, D]
            task: 'pred' or 'recon'
            task_labels: [B, L] tensor with task_id per position
        """
        x = src
        prev_scores = None

        # Create task_labels if not provided (fallback: recon=0)
        if task_labels is None and task is not None:
            task_id = 0  # recon tasks all map to slot 0 (coarse mode)
            task_labels = torch.full((src.shape[0], src.shape[1]), task_id, dtype=torch.long, device=src.device)

        for layer_idx, layer in enumerate(self.layers):
            if self.res_attention:
                layer_out = layer(x, prev=prev_scores, task=task, task_labels=task_labels, attn_mask=attn_mask)
                if isinstance(layer_out, tuple) and len(layer_out) == 2:
                    x, prev_scores = layer_out
                else:
                    x = layer_out
            else:
                x = layer(x, task=task, task_labels=task_labels, attn_mask=attn_mask)

        return x
    
    def get_moe_aux_loss(self) -> Optional[torch.Tensor]:
        """Collect MoE auxiliary loss from all layers"""
        if not self.use_routed_expert:
            return None

        total_aux_loss = None
        for layer in self.layers:
            layer_aux = layer.get_moe_aux_loss()
            if layer_aux is not None:
                if total_aux_loss is None:
                    total_aux_loss = layer_aux
                else:
                    total_aux_loss = total_aux_loss + layer_aux
        return total_aux_loss
    
    def get_routed_l2_loss(self) -> Optional[torch.Tensor]:
        """Collect routed L2 loss from all layers for differential constraint.

        Returns:
            Total routed L2 loss: sum of ||alpha * routed_output||^2 across all layers
        """
        if not self.use_routed_expert:
            return None
        
        total_l2_loss = None
        for layer in self.layers:
            layer_l2 = layer.get_routed_l2_loss()
            if layer_l2 is not None:
                if total_l2_loss is None:
                    total_l2_loss = layer_l2
                else:
                    total_l2_loss = total_l2_loss + layer_l2
        return total_l2_loss
    

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True,
                 activation="gelu", res_attention=False, pre_norm=False,
                 d_task: int = 16,
                 # MoE parameters
                 use_routed_expert: bool = False,
                 num_experts: int = 4,
                 moe_top_k: int = 2,
                 moe_router_fusion_mode: str = 'concat',
                 moe_router_temperature: float = 1.0,
                 # MoE mode: y = y_shared + alpha * dropout(norm(y_routed))
                 use_shared_expert: bool = False):
        super().__init__()
        assert not d_model % n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads
        d_v = d_model // n_heads

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(
            d_model, n_heads, d_k, d_v,
            attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention
        )

        # Add & Norm (Attn)
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # FFN or MoE
        self.use_routed_expert = use_routed_expert
        self.d_model = d_model
        self.d_ff = d_ff
        self.use_shared_expert = use_shared_expert

        # MoE alpha: y = y_shared + alpha * y_routed
        self.moe_alpha = 0.0

        # Shared Expert FFN (single dense FFN)
        if use_shared_expert or not use_routed_expert:
            self.ff_linear1 = nn.Linear(d_model, d_ff, bias=True)
            self.ff_act = nn.GELU() if activation == 'gelu' else nn.ReLU()
            self.ff_dropout_mid = nn.Dropout(dropout)
            self.ff_linear2 = nn.Linear(d_ff, d_model, bias=True)
        else:
            self.ff_linear1 = None
            self.ff_act = None
            self.ff_dropout_mid = None
            self.ff_linear2 = None

        if use_routed_expert:
            from .task_adaptive_moe import TaskAdaptiveMoE
            self.ff = TaskAdaptiveMoE(
                d_model=d_model,
                d_ff=d_ff,
                num_experts=num_experts,
                top_k=moe_top_k,
                d_task=d_task,
                dropout=dropout,
                activation=activation,
                router_fusion_mode=moe_router_fusion_mode,
                use_aux_loss=True,
                router_temperature=moe_router_temperature,
            )
            # Initialize MoE experts from first Shared Expert weights if use_shared_expert
            if use_shared_expert:
                self._init_moe_from_shared_expert()
        else:
            self.ff = None
        
        self._moe_aux_loss = None
        self._routed_l2_loss = None

        # LayerNorm and Dropout for routed output
        if use_shared_expert and use_routed_expert:
            self.moe_output_norm = nn.LayerNorm(d_model)
            self.moe_output_dropout = nn.Dropout(dropout)
        else:
            self.moe_output_norm = None
            self.moe_output_dropout = None

        # Add & Norm (FFN)
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn
        
        self.attn = None
        self.task_token_provider = None  # Injected by TaskTokenManager
    
    def _init_moe_from_shared_expert(self):
        """Initialize routed experts from shared expert weights for smooth warm start."""
        if self.ff is None or self.ff_linear1 is None:
            return
        for expert in self.ff.routed_experts:
            with torch.no_grad():
                expert.net[0].weight.copy_(self.ff_linear1.weight)
                expert.net[0].bias.copy_(self.ff_linear1.bias)
                expert.net[3].weight.copy_(self.ff_linear2.weight)
                expert.net[3].bias.copy_(self.ff_linear2.bias)
    
    def set_moe_alpha(self, alpha: float):
        """Set the alpha value for MoE mode."""
        self.moe_alpha = alpha
        
    def _extract_task_token(self, layer_task_token):
        """Extract task token [B, d_task] from layer_task_token dict for MoE routing"""
        if layer_task_token is None:
            return None
        if isinstance(layer_task_token, dict):
            return layer_task_token["z_task"]
        return layer_task_token
    
    def _prepare_moe_inputs(self, layer_task_token, device, dtype):
        """
        Prepare inputs for MoE forward pass.

        Returns:
            moe_task_token: [B, d_task] or None
            moe_tau: [B, 1] or None
        """
        moe_task_token = None
        moe_tau = None
        if layer_task_token is not None:
            moe_task_token = self._extract_task_token(layer_task_token)
            if isinstance(layer_task_token, dict):
                moe_tau = layer_task_token.get('tau')
            if moe_task_token is not None:
                moe_task_token = moe_task_token.to(device=device, dtype=dtype)
            if moe_tau is not None:
                moe_tau = moe_tau.to(device=device, dtype=dtype)

        return moe_task_token, moe_tau
    
    def forward(self, src: torch.Tensor,
                prev: Optional[torch.Tensor] = None,
                task: Optional[str] = None,
                task_labels: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None):
        """
        Args:
            src: [B, L, D] input tensor
        """
        # Self-attention
        x = src
        if self.pre_norm: 
            x = self.norm_attn(x)
        if self.res_attention:
            x2, attn, scores = self.self_attn(x, x, x, prev, attn_mask=attn_mask)
        else:
            x2, attn = self.self_attn(x, x, x, attn_mask=attn_mask)
            scores = None
        if self.store_attn:
            self.attn = attn
        x = x + self.dropout_attn(x2)
        if not self.pre_norm:
            x = self.norm_attn(x)

        # Extract task_id once from task_labels (used by TaskTokenGenerator and MoE)
        extracted_task_id = None
        if task_labels is not None and task_labels.numel() > 0:
            extracted_task_id = int(task_labels[0, 0].item())

        # Generate layer task token (for MoE routing)
        layer_task_token = None
        if callable(self.task_token_provider):
            provider_task_id = extracted_task_id if extracted_task_id is not None else task
            layer_task_token = self.task_token_provider(
                self, task_id=provider_task_id, current_layer_hidden=x,
                current_layer_attn=self.attn,
            )

        # FFN or MoE
        x_ff_in = self.norm_ffn(x) if self.pre_norm else x

        # Combined MoE mode: y = y_shared + alpha * y_routed
        if self.use_routed_expert and self.use_shared_expert:
            # Compute Shared Expert output
            h_shared = self.ff_linear1(x_ff_in)
            h_shared = self.ff_act(h_shared)
            h_shared = self.ff_dropout_mid(h_shared)
            y_shared = self.ff_linear2(h_shared)

            # Compute Routed Experts output (routed path - gradients flow through)
            moe_task_token, moe_tau = self._prepare_moe_inputs(
                layer_task_token, x.device, x.dtype
            )
            y_routed = self.ff(x_ff_in, z_task=moe_task_token, tau=moe_tau)
            self._moe_aux_loss = self.ff.get_aux_loss()

            # Combine: y = y_shared + alpha * dropout(norm(y_routed))
            alpha = self.moe_alpha
            y_routed_processed = self.moe_output_dropout(self.moe_output_norm(y_routed))
            residual = alpha * y_routed_processed
            ff_out = y_shared + residual

            # Routed L2 loss for differential constraint
            self._routed_l2_loss = residual.pow(2).mean()

        elif self.use_routed_expert:
            # Standard MoE mode (only routed experts, no shared)
            moe_task_token, moe_tau = self._prepare_moe_inputs(
                layer_task_token, x.device, x.dtype
            )
            ff_out = self.ff(x_ff_in, z_task=moe_task_token, tau=moe_tau)
            self._moe_aux_loss = self.ff.get_aux_loss()
        else:
            # Standard FFN
            h = self.ff_linear1(x_ff_in)
            h = self.ff_act(h)
            h = self.ff_dropout_mid(h)
            ff_out = self.ff_linear2(h)
        
        x = x + self.dropout_ffn(ff_out)
        if not self.pre_norm: 
            x = self.norm_ffn(x)
        
        if self.res_attention:
            return x, scores
        return x
    
    def get_moe_aux_loss(self) -> Optional[torch.Tensor]:
        """Get MoE routing auxiliary loss"""
        if self.use_routed_expert and self._moe_aux_loss is not None:
            return self._moe_aux_loss
        return None
    
    def get_routed_l2_loss(self) -> Optional[torch.Tensor]:
        """Get routed L2 loss for differential constraint.

        This loss is computed as ||alpha * routed_output||^2, which constrains
        the scaled routed output to remain small.
        """
        if self.use_routed_expert and self.use_shared_expert and self._routed_l2_loss is not None:
            return self._routed_l2_loss
        return None
    
