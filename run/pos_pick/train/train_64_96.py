import sys
sys.path.append('/home/seleznev.ns/gradwork_ordered/')

import os
import albumentations as A
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from seispick.pos_pick.datasets.transforms import (
    CustomCompose,
    CustomRandCrop,
    CustomCropByTargets,
    CustomRandomShiftLine,
    CustomChangeTargets,
    CustomAddDim,
    CustomToTensor,
    CustomToFloat,
    CustomToFloatNumpy)

from seispick.pos_pick.datasets.dataset import PositionPickingDataSet
from seispick.pos_pick.models.CNN import ConvNet_64_96
from seispick.pos_pick.lightwrappers.lightwrapper import LightWrapper


GPU = 1
EPOCHS = 600
BATCH_SIZE = 8

V = 1442
RADIUS_MIN, RADIUS_MAX = 5, 40
RUN_NAME = f'64_96_({RADIUS_MIN}-{RADIUS_MAX})'

PATH = '/home/seleznev.ns/gradwork_ordered/run/pos_pick/transformed_data/pospick_96'
TRAIN_PATH = os.path.join(PATH, 'train')
TEST_PATH = os.path.join(PATH, 'test')

logger = MLFlowLogger(experiment_name='CNN',
                      run_name=RUN_NAME,
                      save_dir='./mlruns')

# checkpoint_callback = ModelCheckpoint(dirpath='../ckpt/ckpt_CNN/',
#                                       save_top_k=2,
#                                       monitor='val_loss')

trainer = Trainer(accelerator='gpu',
                  devices=[GPU],
                  max_epochs=EPOCHS,
                  logger=logger,)
                #   callbacks=[checkpoint_callback])

train_transforms = CustomCompose([CustomRandCrop(width=96),
                                  CustomRandomShiftLine(min_max_radius=(RADIUS_MIN, RADIUS_MAX), p=0.5),
                                  CustomCropByTargets(V=V, width=64),
                                  CustomChangeTargets(),
                                  CustomAddDim(),
                                  CustomToFloatNumpy(),
                                  A.Compose(
                                            [
                                              A.GaussNoise(var_limit=(1e-5, 1e-6), p=0.3),
                                              A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=0.3),
                                              A.RandomBrightnessContrast(brightness_limit=0.05, 
                                                                         contrast_limit=0.5, 
                                                                         p=0.4)
                                            ]
                                           ),
                                  CustomToTensor(),
                                  CustomToFloat(),
                                ])

test_transforms = CustomCompose([CustomRandCrop(width=96),
                                 CustomRandomShiftLine(min_max_radius=(RADIUS_MIN, RADIUS_MAX), p=0.5),
                                 CustomCropByTargets(V=V, width=64),
                                 CustomChangeTargets(),
                                 CustomAddDim(),
                                 CustomToTensor(),
                                 CustomToFloat(),
                                ])


train_data = PositionPickingDataSet(TRAIN_PATH, transforms=train_transforms)
test_data = PositionPickingDataSet(TEST_PATH, transforms=test_transforms)

train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True, num_workers=8)
test_loader = DataLoader(test_data, BATCH_SIZE, shuffle=False, num_workers=8)

model = LightWrapper(model=ConvNet_64_96,
                     model_settings=dict(),
                     opt=AdamW,
                     opt_settings=dict(lr=1e-3),
                     loss=BCEWithLogitsLoss,
                     loss_settings=dict(),
                     scheduler=None,
                     scheduler_settings=dict())

trainer.fit(model, train_loader, test_loader)
logger.finalize()
