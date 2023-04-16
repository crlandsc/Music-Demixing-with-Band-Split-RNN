from importlib.util import find_spec
import wandb
from utils import pylogger

log = pylogger.get_pylogger(__name__)

def close_loggers() -> None:
    """Makes sure all loggers closed properly (prevents logging failure during multirun)."""

    log.info("Closing loggers...")

    if find_spec("wandb"):  # check if wandb is installed
        import wandb

        if wandb.run:
            log.info("Closing wandb!")
            wandb.finish()


def wandb_login(key):
    wandb.login(key=key)


def wandb_watch_all(wandb_logger, model):
    wandb_logger.watch(model, 'all')