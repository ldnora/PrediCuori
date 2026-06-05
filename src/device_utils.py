import torch
import logging

logger = logging.getLogger(__name__)

try:
    import torch_xla.core.xla_model as xm
    TPU_AVAILABLE = True
except ImportError:
    xm = None
    TPU_AVAILABLE = False


def get_device():
    if TPU_AVAILABLE:
        return xm.xla_device()
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def patch_config_for_device(config: dict, device) -> dict:
    device_str = str(device)
    
    if TPU_AVAILABLE:
        config['pin_memory'] = False
        config['num_workers'] = 4
        config['persistent_workers'] = False
        try:
            import torch_xla
            num_devices = torch_xla.runtime.global_device_count()
        except Exception:
            num_devices = '?'
        logger.info(f'  TPU cores: {num_devices}')

    elif 'cuda' in device_str:
        config['pin_memory'] = True
        config['num_workers'] = 4
        config['persistent_workers'] = True
        logger.info(f'  GPU: {torch.cuda.get_device_name(0)}')

    else:
        config['pin_memory'] = False
        config['num_workers'] = 0
        config['persistent_workers'] = False
        logger.info('  CPU (sem acelerador)')

    logger.info(f'  Device: {device}')
    return config


def optimizer_step(optimizer):
    if TPU_AVAILABLE:
        xm.optimizer_step(optimizer)
        try:
            import torch_xla
            torch_xla.sync()
        except AttributeError:
            xm.mark_step()
    else:
        optimizer.step()


def save_checkpoint(obj, path: str):
    """Drop-in para torch.save() que sincroniza a TPU antes de gravar."""
    if TPU_AVAILABLE:
        xm.save(obj, path)
    else:
        torch.save(obj, path)


def wrap_dataloader(loader, device):
    """Envolve o DataLoader com MpDeviceLoader na TPU."""
    if TPU_AVAILABLE:
        import torch_xla.distributed.parallel_loader as pl
        return pl.MpDeviceLoader(loader, device)
    return loader
