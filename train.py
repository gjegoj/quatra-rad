import re
import sys
import yaml
import glob
import torch
import argparse
import numpy as np
from torch import optim
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

from src.utils.plot import plot, plot_confusion_matrix
from src.models import resnet18, resnet34, effnet
from src.dataset.dataset import JapanItemsDataset
from src.dataset.transforms import train_transform, val_transform

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(ROOT)) if str(ROOT) not in sys.path else None

def increment_path(path, exist_ok=False, sep='_', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(
            ''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=ROOT /
                        'src/cfg/default_cfg.yaml', help='configuration file path')

    args = parser.parse_args()
    return args


class Lightning(pl.LightningModule):
    def __init__(self, model, cfg):
        super(Lightning, self).__init__()

        self.model = model
        self.cfg = cfg

    def forward(self, x):
        output = self.model.forward(x)
        return output

    def loss_function(self, logits, targets):
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        return criterion(logits, targets)

    def _plot(self, img, targets, preds, n_images):
        return plot(img, targets, preds, n_images)

    def _plot_confusion_matrix(self, targets, preds):
        return plot_confusion_matrix(targets, preds)

    def _acc(self, targets, preds):
        return (preds == targets).sum().float() / (float(targets.shape[0]))

    def training_step(self, batch, batch_idx):
        x, targets = batch
        logits = self.forward(x)
        train_loss = self.loss_function(logits, targets)

        preds = torch.argmax(torch.softmax(logits, dim=1), axis=1)

        train_acc = self._acc(targets, preds).detach().cpu().numpy()

        current_lr = self.get_current_lr().to(x.device)

        with torch.no_grad():
            if batch_idx == 0:

                self.logger.experiment.add_images('train/pred',
                                                  self._plot(
                                                            x.detach().cpu().numpy(),
                                                            targets.detach().cpu().numpy(),
                                                            preds.detach().cpu().numpy(),
                                                            32),
                                                  global_step=self.current_epoch
                                                  )

        return {'loss': train_loss,
                'log': {'train_loss': train_loss, 'train_acc': train_acc, 'current_lr': current_lr}}

    def validation_step(self, val_batch, batch_idx):
        x, targets = val_batch
        logits = self.forward(x)
        val_loss = self.loss_function(logits, targets)

        preds = torch.argmax(torch.softmax(logits, dim=1), axis=1)

        val_acc = self._acc(targets, preds).detach().cpu().numpy()

        with torch.no_grad():
            if batch_idx == 0:

                self.logger.experiment.add_images('val/pred',
                                                  self._plot(
                                                            x.detach().cpu().numpy(),
                                                            targets.detach().cpu().numpy(),
                                                            preds.detach().cpu().numpy(),
                                                            32),
                                                  global_step=self.current_epoch
                                                  )

        return {'val_loss': val_loss, 
                'val_acc': val_acc, 
                'targets': targets.detach().cpu().numpy(), 
                'preds': preds.detach().cpu().numpy()}

    def validation_epoch_end(self, val_outputs):

        def _get_average(outputs, key):
            assert isinstance(outputs, list)
            assert isinstance(key, str)

            metric_mean = 0
            for output in outputs:
                metric = output[key]
                metric_mean += metric
            return metric_mean / len(outputs)

        val_loss_mean = _get_average(val_outputs, 'val_loss')
        soft_acc_mean = _get_average(val_outputs, 'val_acc')

        all_targets = np.array([])
        all_preds = np.array([])

        for output in val_outputs:
            all_targets = np.concatenate((all_targets, output['targets']))
            all_preds = np.concatenate((all_preds, output['preds']))

        with torch.no_grad():

                self.logger.experiment.add_images('val/confusion_matrix',
                                                  self._plot_confusion_matrix(
                                                                            all_targets,
                                                                            all_preds),
                                                  global_step=self.current_epoch
                                                  )

        return {'progress_bar': {'val_loss': val_loss_mean}, 
                'log': {'val_loss': val_loss_mean, 'val_acc': soft_acc_mean}}

    def train_dataloader(self):
        dataset = JapanItemsDataset(transform=train_transform, 
                                    **self.cfg['TRAIN']['DATASET']['ARGS'])

        return DataLoader(dataset, **self.cfg['TRAIN']['DATALOADER']['ARGS'])

    def val_dataloader(self):
        dataset = JapanItemsDataset(transform=val_transform, 
                                    **self.cfg['VAL']['DATASET']['ARGS'])

        return DataLoader(dataset, **self.cfg['VAL']['DATALOADER']['ARGS'])

    def configure_optimizers(self):
        optimizer = optim.AdamW(params=filter(
            lambda p: p.requires_grad, self.model.parameters()), **self.cfg['OPTIMIZER']['ARGS'])

        if self.cfg['SCHEDULER']['NAME'] == 'CyclicLR':
            scheduler = lr_scheduler.CyclicLR(optimizer, **self.cfg['SCHEDULER']['ARGS'])

        elif self.cfg['SCHEDULER']['NAME'] == 'CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, **self.cfg['SCHEDULER']['ARGS'])

        else:
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **self.cfg['SCHEDULER']['ARGS'])

        return [optimizer], [scheduler]

    def get_current_lr(self):
        return torch.Tensor([np.mean([params['lr'] for opt in self.trainer.optimizers for params in opt.param_groups])])


def reproducibility(cfg):
    torch.backends.cudnn.benchmark = cfg['CUDNN']['BENCHMARK']
    torch.backends.cudnn.deterministic = cfg['CUDNN']['DETERMINISTIC']
    if cfg['REPRODUCIBILITY']['ENABLE']:
        torch.manual_seed(cfg['REPRODUCIBILITY']['TORCH_SEED'])
        np.random.seed(cfg['REPRODUCIBILITY']['NUMPY_SEED'])


if __name__ == '__main__':

    args = parse_args()

    with open(args.cfg, errors='ignore') as f:
        cfg = yaml.safe_load(f)

    reproducibility(cfg)

    save_dir = increment_path(Path(cfg['PROJECT']))

    # Make dir with logs
    logger_train = TensorBoardLogger(
        save_dir=str(save_dir),
        name='logs',
        version=str(datetime.now().strftime("%d_%H_%M_%S_")) + (cfg['MODEL']['NAME'] if cfg['NAME'] == '' else cfg['NAME']))

    # Make dir with weights
    (save_dir / 'weights').mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
                                        filepath=str(save_dir / 'weights'), 
                                        **cfg['CHECKPOINTS']['ARGS'])

    # Save hyperparameters
    with open(str(save_dir) + '/hyp.yaml', 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)

    base_model = resnet18.Model()
    # base_model = effnet.Model()


    if cfg['MODEL']['PRETRAINED']['ENABLE']:
        print(
            f"Loading from checkpoint: {cfg['MODEL']['PRETRAINED']['WEIGHTS']}")
        model_weights = torch.load(
            cfg['MODEL']['PRETRAINED']['WEIGHTS'], map_location='cpu')['state_dict']
        model_weights = {key[6:]: model_weights[key]
                         for key in model_weights.keys()}
        base_model.load_state_dict(
            model_weights, strict=cfg['MODEL']['PRETRAINED']['STRICT_LOAD'])
        print('Weights were loaded!')

    model = Lightning(base_model, cfg)

    trainer = Trainer(
        logger=logger_train,
        checkpoint_callback=checkpoint_callback,
        gpus=cfg['TRAINER']['GPU'],
        min_epochs=cfg['TRAINER']['MIN_EPOCHS'],
        max_epochs=cfg['TRAINER']['MAX_EPOCHS'],
        check_val_every_n_epoch=cfg['TRAINER']['CHECK_VAL_EVERY_N_EPOCH'],
        show_progress_bar=cfg['TRAINER']['SHOW_PROGRESS_BAR'],
    )

    trainer.fit(model)
