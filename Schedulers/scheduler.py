from Schedulers.cosine_scheduler import CosineAnnealingLR
from Schedulers.lambda_scheduler import LambdaLR
from Schedulers.step_scheduler import StepLR


def constant_lr_factor(_):
    """Pickle-safe constant LR multiplier used by no-op LambdaLR schedulers."""
    return 1.0


def cosine_scheduler(optimizer, args):
    """Cosine annealing over the full training run."""
    return CosineAnnealingLR(
        optimizer,
        T_max=args.num_steps,
    )


def step_scheduler(optimizer, args):
    """Step decay: multiply LR by gamma every lr_step_size steps."""
    return StepLR(
        optimizer,
        step_size=getattr(args, "lr_step_size", 10000),
        gamma=getattr(args, "lr_gamma", 0.5),
    )


def lambda_scheduler(optimizer, args):
    """LambdaLR with a constant factor of 1.0 so learning rate stays fixed."""
    return LambdaLR(optimizer, lr_lambda=constant_lr_factor)


def none_scheduler(optimizer, args):
    """Alias for a no-op scheduler used by the notebook defaults."""
    return LambdaLR(optimizer, lr_lambda=constant_lr_factor)


schedulers = {
    "cosine":  cosine_scheduler,
    "step":    step_scheduler,
    "lambda":  lambda_scheduler,
    "none":    none_scheduler,
}
