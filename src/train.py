import argparse

from typing import List, Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from typing import Tuple
import shutil

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler

from data import SourceSeparationDataset, collate_fn
from model import BandSplitRNN, PLModel

import logging
import traceback

import utils

from pytorch_lightning.loggers import WandbLogger

log = logging.getLogger(__name__)


def initialize_loaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader]:
    """
    Initializes train and validation dataloaders from configuration file.
    """
    train_dataset = SourceSeparationDataset(
        **cfg.train_dataset,
    )
    train_loader = DataLoader(
        train_dataset,
        **cfg.train_loader,
        collate_fn=collate_fn
    )
    if hasattr(cfg, 'val_dataset'):
        val_dataset = SourceSeparationDataset(
            **cfg.val_dataset,
        )
        val_loader = DataLoader(
            val_dataset,
            **cfg.val_loader,
            collate_fn=collate_fn
        )
    else:
        val_loader = None
    return (
        train_loader,
        val_loader
    )


def initialize_featurizer(
        cfg: DictConfig
) -> Tuple[nn.Module, nn.Module]:
    """
    Initializes direct and inverse featurizers for audio.
    """
    featurizer = instantiate(
        cfg.featurizer.direct_transform,
    )
    inv_featurizer = instantiate(
        cfg.featurizer.inverse_transform,
    )
    return featurizer, inv_featurizer


def initialize_augmentations(
        cfg: DictConfig
) -> nn.Module:
    """
    Initializes augmentations.
    """
    augs = instantiate(cfg.augmentations)
    augs = nn.Sequential(*augs.values())
    return augs


def initialize_model(
        cfg: DictConfig
) -> Tuple[nn.Module, Optimizer, lr_scheduler._LRScheduler]:
    """
    Initializes model from configuration file.
    """
    # initialize model
    model = BandSplitRNN(
        **cfg.model
    )
    # initialize optimizer
    if hasattr(cfg, 'opt'):
        opt = instantiate(
            cfg.opt,
            params=model.parameters()
        )
    else:
        opt = None
    # initialize scheduler
    if hasattr(cfg, 'sch'):
        if hasattr(cfg.sch, '_target_'):
            # other than LambdaLR
            sch = instantiate(
                cfg.sch,
                optimizer=opt
            )
        else:
            # if LambdaLR
            lr_lambda = lambda epoch: (
                cfg.sch.alpha ** (cfg.sch.warmup_step - epoch)
                if epoch < cfg.sch.warmup_step
                else cfg.sch.gamma ** (epoch - cfg.sch.warmup_step)
            )
            sch = torch.optim.lr_scheduler.LambdaLR(
                optimizer=opt,
                lr_lambda=lr_lambda
            )
    else:
        sch = None
    return model, opt, sch


def initialize_utils(
        cfg: DictConfig
):
    # change model and logs saving directory to logging directory of hydra
    if HydraConfig.instance().cfg is not None:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        save_dir = hydra_cfg['runtime']['output_dir']
        cfg.logger.tensorboard.save_dir = save_dir + cfg.logger.tensorboard.save_dir # update tensorboard save_dir
        # cfg.logger.wandb.save_dir = save_dir + cfg.logger.wandb.save_dir # update wandb save_dir
        if hasattr(cfg.callbacks, 'model_ckpt'):
            cfg.callbacks.model_ckpt.dirpath = save_dir + cfg.callbacks.model_ckpt.dirpath
    # delete early stopping if there is no validation dataset
    if not hasattr(cfg, 'val_dataset') and hasattr(cfg.callbacks, 'early_stop'):
        del cfg.callbacks.early_stop

    # initialize loggers
    logger: List = []
    if "logger" in cfg:
        # logger = instantiate(cfg.logger)
        for _, lg_conf in cfg["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(instantiate(lg_conf))
                # if lg_conf._target_ == "pytorch_lightning.loggers.WandbLogger":
                #     wandb_logger = WandbLogger(**lg_conf)
                #     logger.append(wandb_logger)
                # else:
                #     logger.append(instantiate(lg_conf))

        # Initialize wandb logger
        for wandb_logger in [l for l in logger if isinstance(l, WandbLogger)]:
                utils.wandb_login(key=cfg.wandb_api_key)
                # utils.wandb_watch_all(wandb_logger, model) # TODO buggy
                break
    
    # initialize callbacks
    callbacks = list(instantiate(cfg.callbacks).values())

    return logger, callbacks


@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(
        # device: str,
        cfg: DictConfig
) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    pl.seed_everything(42, workers=True)

    log.info(OmegaConf.to_yaml(cfg))

    log.info("Initializing loaders, featurizers.")
    train_loader, val_loader = initialize_loaders(cfg)
    featurizer, inverse_featurizer = initialize_featurizer(cfg)
    augs = initialize_augmentations(cfg)

    log.info("Initializing model, optimizer, scheduler.")
    model, opt, sch = initialize_model(cfg)

    log.info("Initializing Lightning logger and callbacks.")
    logger, callbacks = initialize_utils(cfg)

    log.info("Initializing Lightning modules.")
    plmodel = PLModel(
        model,
        featurizer, inverse_featurizer,
        augs,
        opt, sch,
        cfg
    )
    trainer = pl.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )
    
    # if cfg.chkp_path != None:
    #     # Load pl checkpoint into model
    #     plmodel = plmodel.load_from_checkpoint(cfg.chkp_path)

    log.info("Starting training...")
    try:
        trainer.fit(
            plmodel,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=cfg.ckpt_path
        )
    except Exception as e:
        log.error(traceback.format_exc())

    log.info("Training finished!")

    if cfg.trainer.fast_dev_run:
        hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
        shutil.rmtree(hydra_cfg['runtime']['output_dir'])

    # Print path to best checkpoint
    log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = cfg.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '-d',
    #     '--device',
    #     type=str,
    #     required=False,
    #     default='cpu',
    #     help="Device name - either 'cuda', or 'cpu'."
    # )
    # parser.add_argument(
    #     '-c',
    #     '--ckpt-path',
    #     type=str,
    #     required=False,
    #     default=None,
    #     help="Path to .ckpt file to train model from a checkpoint (including .ckpt extention)"
    # )
    # args = vars(parser.parse_args())

    # train(args['device'])
    train()
