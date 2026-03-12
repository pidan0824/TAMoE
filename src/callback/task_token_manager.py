"""
Task Token Manager Callback
============================
Key Components:
1. StateGate: global_desc -> (alpha_view, tau)
2. QueryBuilder: Q = q_task + alpha_view * q_view
3. CR (Contextual Representation): z_context = Attn(Q, hidden_tokens)
4. VWR (Variable-Wise Representation): z_vise = beta * (periodic + spectral + stat)

Token Convention:
- z_task [B, d_task]: Final task control token for MoE routing
- tau [B, 1]: Dynamic temperature for router from StateGate
- global_desc [B, global_desc_dim]: Statistical features from backbone hidden states
"""

__all__ = ['TaskTokenManager']
import torch
from typing import Optional
from pathlib import Path
import json

from .core import Callback
from .multi_task_callback import TASK_ID_MAP, build_view_meta_batch
from ..models.task_token import extract_global_desc_from_layer


class TaskTokenManager(Callback):
    """
    Task Token Manager Callback:
    1. Initialize and manage TaskTokenGenerator
    2. Inject layer task_token_provider to each backbone layer
    """
    
    def __init__(
        self,
        use_task_token: bool = False,
        d_task: int = 16,
        d_model: int = 128,
        view_dim: int = 8,
        global_desc_dim: int = 16,
        use_fine_grained_task_id: bool = False,
        vwr_beta: float = 0.5,
        nhead: int = 4,
        dropout: float = 0.1,
        use_cr: bool = True,
        use_vwr: bool = True,
        use_periodic: bool = True,
        use_spectral: bool = True,
        use_stat: bool = True,
        use_global_desc: bool = True,
        use_learnable_vwr: bool = False,
        verbose: bool = False,
        **unused_kwargs,
    ):
        self.use_task_token = use_task_token
        self.d_task = d_task
        self.d_model = d_model
        self.view_dim = view_dim
        self.global_desc_dim = global_desc_dim
        self.use_fine_grained_task_id = use_fine_grained_task_id
        self.vwr_beta = vwr_beta
        self.nhead = nhead
        self.dropout = dropout
        self.use_cr = use_cr
        self.use_vwr = use_vwr
        self.use_periodic = use_periodic
        self.use_spectral = use_spectral
        self.use_stat = use_stat
        self.use_global_desc = use_global_desc
        self.use_learnable_vwr = use_learnable_vwr
        self.verbose = verbose
        
        self.task_token_gen = None
        self._params_registered = False
    
    def _ensure_all_params_in_optimizer(self, force_requires_grad: bool = False):
        """Ensure all task token generator params are in optimizer.
        
        Args:
            force_requires_grad: If True, set requires_grad=True on all params
                before checking. Used after optimizer rebuild (resets the
                registration flag so params are re-checked).
        """
        if not self.use_task_token or self.task_token_gen is None:
            return
        
        if self._params_registered and not force_requires_grad:
            return
        
        if self.learner.opt is None:
            return
        
        if force_requires_grad:
            for p in self.task_token_gen.parameters():
                p.requires_grad = True
        
        opt_param_ids = {id(p) for group in self.learner.opt.param_groups for p in group['params']}
        new_params = [p for p in self.task_token_gen.parameters() 
                      if p.requires_grad and id(p) not in opt_param_ids]
        
        if new_params:
            base_group = self.learner.opt.param_groups[0] if self.learner.opt.param_groups else {}
            base_lr = base_group.get('lr', 1e-4)
            
            new_group = {'params': new_params, 'lr': base_lr}
            for key in ['betas', 'momentum', 'weight_decay', 'eps', 'amsgrad',
                        'alpha', 'centered', 'max_lr', 'initial_lr', 'min_lr']:
                if key in base_group:
                    new_group[key] = base_group[key]
            
            self.learner.opt.add_param_group(new_group)
            total_params = sum(p.numel() for p in new_params)
            if self.verbose:
                print(f'[TaskToken] Added to optimizer: {total_params:,} params')
        
        self._params_registered = True
    
    def _get_backbone_d_model(self) -> int:
        """Get backbone's hidden dimension."""
        actual_model, backbone = self._get_model_and_backbone()
        if backbone is not None:
            return getattr(backbone, 'd_model', 128)
        return getattr(actual_model, 'd_model', 128)
    
    def _initialize_task_token_generator(self):
        """Initialize TaskTokenGenerator."""
        if self.task_token_gen is not None:
            return
        
        if not self.use_task_token:
            return
        
        model = self.learner.model
        device = next(model.parameters()).device
        actual_model, backbone = self._get_model_and_backbone()
        
        existing_tt_gen = getattr(model, 'task_token_generator', None) or getattr(actual_model, 'task_token_generator', None)

        d_model = self._get_backbone_d_model()

        if existing_tt_gen is not None:
            self.task_token_gen = existing_tt_gen.to(device)
            if self.verbose:
                print(f'[TaskToken] Using existing encoder')
        else:
            from ..models.task_token import TaskTokenGenerator
            self.task_token_gen = TaskTokenGenerator(
                d_task=self.d_task,
                d_model=d_model,
                view_dim=self.view_dim,
                global_desc_dim=self.global_desc_dim,
                use_fine_grained_task_id=self.use_fine_grained_task_id,
                vwr_beta=self.vwr_beta,
                nhead=self.nhead,
                dropout=self.dropout,
                use_cr=self.use_cr,
                use_vwr=self.use_vwr,
                use_periodic=self.use_periodic,
                use_spectral=self.use_spectral,
                use_stat=self.use_stat,
                use_global_desc=self.use_global_desc,
                use_learnable_vwr=self.use_learnable_vwr,
            ).to(device)
        
        self._inject_task_token_provider_to_layers()
        
        total_params = sum(p.numel() for p in self.task_token_gen.parameters())
        if self.verbose:
            print(f'[TaskToken] {total_params:,} params (trainable)')
    
    def _compute_task_token(
        self,
        layer_idx: int,
        task_id,
        current_layer_hidden: Optional[torch.Tensor] = None,
        current_layer_attn: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute task token (z_task) and temperature (tau) for a single layer.

        This is the core logic called by the thin ``layer_provider`` wrapper
        that gets injected into each TransformerLayer.
        """
        device = next(self.task_token_gen.parameters()).device

        if task_id is None:
            task_id = 0
        elif isinstance(task_id, str):
            task_id = TASK_ID_MAP.get(task_id, 0)

        forward_kwargs = getattr(self.learner, 'forward_kwargs', None) or {}
        padding_mask = forward_kwargs.get('padding_mask')

        task_state = getattr(self.learner, '_task_state', None) or {}
        view_meta = task_state.get('view_meta')

        xb = self.learner.xb
        if isinstance(xb, (tuple, list)):
            xb = xb[0]
        x_input = xb.to(device)

        # xb shape: [B, num_patch, n_vars, patch_len] or [B, seq_len, n_vars]
        B = xb.shape[0]
        n_vars = xb.shape[2] if len(xb.shape) >= 3 else 1

        hidden_tokens = None
        attn_weights = None
        if current_layer_hidden is not None:
            B_flat, L, D = current_layer_hidden.shape
            if B_flat != B * n_vars and B > 0 and B_flat % B == 0:
                n_vars = B_flat // B
            hidden_tokens = current_layer_hidden.view(B, n_vars, L, D).mean(dim=1)

            if current_layer_attn is not None:
                H = current_layer_attn.shape[1]
                attn_weights = current_layer_attn.view(B, n_vars, H, L, L).mean(dim=1)

        global_desc = extract_global_desc_from_layer(
            hidden=hidden_tokens,
            attention=attn_weights,
            padding_mask=padding_mask,
            global_desc_dim=self.global_desc_dim,
        )

        if view_meta is None:
            view_meta = build_view_meta_batch(
                batch_size=B, task_name='PM', mask_ratio=0.4, device=device,
            )

        task_id_tensor = torch.full((B,), task_id, dtype=torch.long, device=device)

        outputs = self.task_token_gen(
            task_id=task_id_tensor,
            view_meta=view_meta,
            global_desc=global_desc,
            hidden_tokens=hidden_tokens,
            x_input=x_input,
            padding_mask=padding_mask,
            return_all=True,
        )

        self._last_z_task = outputs['z_task']
        self._last_task_id = task_id

        return {'z_task': outputs['z_task'], 'tau': outputs['tau']}

    def _inject_task_token_provider_to_layers(self):
        """Inject task_token_provider to each backbone layer."""
        actual_model, backbone = self._get_model_and_backbone()

        from ..models.tamoe_backbone import TransformerLayer

        def layer_provider(layer_module, task_id=None,
                           current_layer_hidden=None, current_layer_attn=None):
            return self._compute_task_token(
                layer_idx=layer_module._layer_idx,
                task_id=task_id,
                current_layer_hidden=current_layer_hidden,
                current_layer_attn=current_layer_attn,
            )

        import re
        injected = 0
        for name, module in actual_model.named_modules():
            if isinstance(module, TransformerLayer):
                match = re.search(r"layers\.(\d+)", name)
                if match is None:
                    continue
                idx = int(match.group(1))

                if not module.store_attn:
                    module.store_attn = True
                    module.attn = None

                module._layer_idx = idx
                module.task_token_provider = layer_provider
                injected += 1

        if self.verbose:
            print(f'[TaskToken] Injected task_token_provider to {injected} TransformerLayer(s)')
    
    def before_fit(self):
        """Initialize before training."""
        self._initialize_task_token_generator()
        self._ensure_all_params_in_optimizer()
    
    def before_test(self):
        """Initialize before testing."""
        self._initialize_task_token_generator()

        if hasattr(self, '_test_tt_state_dict') and self._test_tt_state_dict is not None:
            target_state = self.task_token_gen.state_dict()
            filtered_state = {
                k: v for k, v in self._test_tt_state_dict.items()
                if k in target_state and isinstance(v, torch.Tensor) 
                and isinstance(target_state[k], torch.Tensor) and v.shape == target_state[k].shape
            }
            if filtered_state:
                self.task_token_gen.load_state_dict(filtered_state, strict=False)
                if self.verbose:
                    print(f"[TaskToken] Loaded {len(filtered_state)} params for testing")
        
        if self.task_token_gen is not None:
            for param in self.task_token_gen.parameters():
                param.requires_grad = False
    
    def before_epoch(self):
        self._ensure_all_params_in_optimizer()
    
    def before_forward(self):
        self._last_z_task = None
    
    def save_task_token_weights(self, fname: str, path: str):
        """
        Save task token generator weights (called by SaveModelCB via extra_save_fn).
        
        This method is designed to be passed as extra_save_fn to SaveModelCB,
        ensuring weights are saved whenever the best model is saved.
        """
        if self.task_token_gen is None:
            return
        
        weight_path = Path(path) / f'{fname}_task_token.pth'
        torch.save(self.task_token_gen.state_dict(), weight_path)
        if self.verbose:
            print(f"[TaskToken] Best weights saved: {weight_path}")
        
        config_path = Path(path) / f'{fname}_task_token.json'
        with open(config_path, 'w') as f:
            json.dump(self.task_token_gen.config, f, indent=2)

    def after_fit(self):
        """Save final weights as backup if no best was saved by SaveModelCB."""
        if self.task_token_gen is None:
            return

        save_path = getattr(self.learner, 'save_path', None)
        save_name = getattr(self.learner, 'save_pretrained_model', None) or \
                    getattr(self.learner, 'save_finetuned_model', None)

        if save_path and save_name:
            weight_file = Path(save_path) / f'{save_name}_task_token.pth'
            if not weight_file.exists():
                self.save_task_token_weights(save_name, save_path)
                if self.verbose:
                    print(f"[TaskToken] Warning: No best model saved, saved final weights")
            else:
                saved_state = torch.load(weight_file, map_location='cpu')
                self.task_token_gen.load_state_dict(saved_state, strict=True)
                if self.verbose:
                    print(f"[TaskToken] Loaded best weights: {weight_file}")

