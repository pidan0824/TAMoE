"""
MoE-related callbacks for Task-Adaptive Mixture of Experts.

Includes:
- MoEAlphaScheduleCB: Schedules alpha y = y_shared + alpha * LN(y_routed)
- MoEAuxLossCB: Load balancing auxiliary loss
- MoERoutedL2CB: L2 regularization on routed experts output
"""

__all__ = ['MoEAlphaScheduleCB', 'MoEAuxLossCB', 'MoERoutedL2CB']

from .core import Callback


class MoEAlphaScheduleCB(Callback):
    """
    Callback to schedule alpha for MoE: y = y_shared + alpha * LN(y_routed)

    Args:
        schedule: Schedule type ('linear', 'fixed', or 'plateau')
        alpha_start: Starting alpha value
        alpha_end: Ending alpha value (ignored when schedule='fixed')
        warmup_epochs: Epochs to hold alpha_start before ramping (linear only)
        plateau_start: Epoch to start ramping alpha (plateau schedule only)
    """

    def __init__(
        self,
        schedule: str = 'linear',
        alpha_start: float = 0.1,
        alpha_end: float = 0.15,
        warmup_epochs: int = 0,
        plateau_start: int = 50,
    ):
        super().__init__()
        assert schedule in ['linear', 'fixed', 'plateau'], \
            f"schedule must be 'linear', 'fixed', or 'plateau', got {schedule}"
        self.schedule = schedule
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end if schedule != 'fixed' else alpha_start
        self.warmup_epochs = warmup_epochs
        self.plateau_start = plateau_start
        self.current_alpha = alpha_start

    def before_fit(self):
        """Initialize at training start."""
        self.total_epochs = self.n_epochs
        print(f"[MoEAlpha] Schedule: {self.schedule}")
        if self.schedule == 'fixed':
            print(f"[MoEAlpha] Alpha: fixed at {self.alpha_start}")
        elif self.schedule == 'plateau':
            print(f"[MoEAlpha] Alpha: {self.alpha_start} (epochs 0-{self.plateau_start-1})")
            print(f"[MoEAlpha] Alpha: {self.alpha_start} -> {self.alpha_end} (epochs {self.plateau_start}-{self.total_epochs-1})")
        else:
            print(f"[MoEAlpha] Alpha: {self.alpha_start} -> {self.alpha_end} over {self.total_epochs} epochs")
        if self.warmup_epochs > 0 and self.schedule not in ['fixed', 'plateau']:
            print(f"[MoEAlpha] Warmup: {self.warmup_epochs} epochs (alpha={self.alpha_start})")

        self._set_alpha(self.alpha_start)

    def before_epoch(self):
        """Update alpha at the start of each epoch."""
        epoch = self.epoch
        alpha = self._compute_alpha(epoch)
        self.current_alpha = alpha
        self._set_alpha(alpha)

        if epoch % 10 == 0:
            print(f"[MoEAlpha] Epoch {epoch}: alpha = {alpha:.4f}")

    def _compute_alpha(self, epoch: int) -> float:
        """Compute alpha value for given epoch (0-indexed)."""
        if self.schedule == 'fixed':
            return self.alpha_start

        if self.schedule == 'plateau':
            if epoch < self.plateau_start:
                return self.alpha_start
            ramp_epochs = self.total_epochs - self.plateau_start
            if ramp_epochs <= 0:
                return self.alpha_end
            progress = min((epoch - self.plateau_start) / ramp_epochs, 1.0)
            return self.alpha_start + (self.alpha_end - self.alpha_start) * progress

        if epoch < self.warmup_epochs:
            return self.alpha_start

        effective_epoch = epoch - self.warmup_epochs
        effective_total = self.total_epochs - self.warmup_epochs

        if effective_total <= 0:
            return self.alpha_end

        progress = min(effective_epoch / effective_total, 1.0)
        return self.alpha_start + (self.alpha_end - self.alpha_start) * progress

    def _set_alpha(self, alpha: float):
        """Set alpha value for all MoE layers in the model."""
        from ..models.tamoe_backbone import TransformerLayer

        for module in self.model.modules():
            if isinstance(module, TransformerLayer) and module.use_shared_expert:
                module.set_moe_alpha(alpha)

    def after_fit(self):
        """Log final alpha value."""
        print(f"[MoEAlpha] Training complete. Final alpha = {self.current_alpha:.4f}")


class MoEAuxLossCB(Callback):
    """
    Add MoE load balancing auxiliary loss to total loss for optimization.

    Load balancing loss ensures experts are used uniformly, preventing
    expert collapse during training.

    """
    def __init__(self, aux_loss_weight: float = 0.0001):
        """
        Args:
            aux_loss_weight: Scaling factor for aux_loss (default 0.0001)
        """
        super().__init__()
        self.aux_loss_weight = aux_loss_weight

    def _is_shared_only_mode(self, backbone) -> bool:
        """Check if any MoE layer is in shared_only mode (routing not used)."""
        encoder = getattr(backbone, 'transformer', None)
        if encoder is None:
            return False
        for layer in encoder.layers:
            if hasattr(layer, 'ff') and getattr(layer.ff, 'aggregation_mode', None) == 'shared_only':
                return True
        return False

    def before_backward(self):
        backbone = self._get_backbone()
        if backbone is None:
            return

        if self._is_shared_only_mode(backbone):
            return

        aux_loss = backbone.get_moe_aux_loss()
        if aux_loss is not None and aux_loss.requires_grad:
            self.learner.loss = self.learner.loss + self.aux_loss_weight * aux_loss


class MoERoutedL2CB(Callback):
    """
    L2 regularization on routed output for differential constraint.

    Loss added: weight * ||alpha * routed_output||^2
    Keeps Routed Experts as a small correction relative to the Shared Expert.
    """

    def __init__(self, weight: float = 1e-4):
        super().__init__()
        self.weight = weight

    def before_fit(self):
        """Log configuration at training start."""
        print(f"[MoERoutedL2] Routed L2 regularization weight: {self.weight}")

    def before_backward(self):
        """Add routed L2 loss to total loss before backward pass."""
        backbone = self._get_backbone()
        if backbone is None:
            return

        routed_l2 = backbone.get_routed_l2_loss()
        if routed_l2 is not None and routed_l2.requires_grad:
            self.learner.loss = self.learner.loss + self.weight * routed_l2