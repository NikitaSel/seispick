import pytorch_lightning as pl
import tempfile
import torch
import os

class LightWrapper(pl.LightningModule):
    def __init__(self, model, model_settings,
                       opt, opt_settings,
                       loss, loss_settings,
                       scheduler=None, scheduler_settings=None):
        super().__init__()
        self.opt = opt
        self.loss = loss
        self.scheduler = scheduler
        self.save_hyperparameters(dict(model_name = model.__name__,
                                       model_settings = model_settings,
                                       opt_name = opt.__name__,
                                       opt_settings = opt_settings,
                                       loss_name = loss.__name__,
                                       loss_settings = loss_settings,
                                       scheduler_name = scheduler.__name__ if scheduler else None,
                                       scheduler_settings = scheduler_settings
                                       ))

        self.best_score = 0
        self.best_loss = None
        self.model = model(**model_settings)
        self.loss = loss(**loss_settings)

        # checkpoint = torch.load('./ckpt_CNN/CNN_64_64.ckpt')
        # self.model.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, x):
        return self.model.forward(x)

    def on_train_start(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'model_summary.txt'), 'w') as f:
                f.write(str(pl.utilities.model_summary.ModelSummary(self)))
            self.logger.experiment.log_artifact(self.logger.run_id, os.path.join(tmpdir, 'model_summary.txt'))

    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch['input'], batch['target']
        y_pred = self.model(X_batch)
        y_pred = y_pred.sum(axis=(1, 2))

        loss = self.loss(y_pred, y_batch)
        score = self._get_score(y_pred, y_batch)

        self.log('train_loss', loss, on_epoch=True, on_step=False)
        self.log('train_score', score, on_epoch=True, on_step=False)

        return {'loss': loss, 'score': score}

    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch['input'], batch['target']
        y_pred = self.model(X_batch)
        y_pred = y_pred.sum(axis=(1, 2))

        loss = self.loss(y_pred, y_batch)
        score = self._get_score(y_pred, y_batch)
         
        self.log('val_loss', loss, on_epoch=True, on_step=False)
        self.log('val_score', score, on_epoch=True, on_step=False)

        if (self.best_loss is None) or loss <= self.best_loss:
            self.best_loss = loss
            torch.save({
                'model_state_dict': self.model.state_dict(),
                }, f'../ckpt/ckpt_{self.model.__class__.__name__}_(!1819)')

        return {'val_loss': loss, 'val_score': score}

    def configure_optimizers(self):
        optimizer = self.opt(self.model.parameters(), **self.hparams['opt_settings'])
        if self.scheduler is None:
            return optimizer
        scheduler = self.scheduler(optimizer, **self.hparams['scheduler_settings'])
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def _get_score(self, y_pred, y_true):
        score = (y_pred.argmax(dim=1) - y_true.argmax(dim=1)) ** 2
        score = score.to(torch.float).mean()
        score = score.sqrt()
        return score
