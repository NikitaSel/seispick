import sys
sys.path.append('/home/seleznev.ns/gradwork_ordered/')

import os
import albumentations as A
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from seispick.time_pick.datasets.transforms import (
	FlattenTargets,
	InputsToMatrix,
	ToTensor,
	ToFloat,
)

from seispick.time_pick.datasets.dataset import TimePickingDataSet
from seispick.time_pick.models.Unet import UNet
from seispick.time_pick.lightwrappers.lightwrapper import LightWrapper


GPU = 0
EPOCHS = 50
BATCH_SIZE = 15

RUN_NAME = 'Unet_!1819'

PATH = '/home/seleznev.ns/gradwork_ordered/run/time_pick/transformed_data/'
TRAIN_PATH = os.path.join(PATH, 'train', 'data')
TEST_PATH = os.path.join(PATH, 'test', 'data')

logger = MLFlowLogger(experiment_name='Unet',
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

train_transforms = A.Compose([FlattenTargets(epsilon=0.01),
							  InputsToMatrix(size=32),
							  ToTensor(),
							  ToFloat(),
                            ])

test_transforms = A.Compose([FlattenTargets(epsilon=0.01),
							  InputsToMatrix(size=32),
							  ToTensor(),
							  ToFloat(),
                            ])

train_data = TimePickingDataSet(TRAIN_PATH, transforms=train_transforms)
test_data = TimePickingDataSet(TEST_PATH, transforms=test_transforms)

train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True, num_workers=8)
test_loader = DataLoader(test_data, BATCH_SIZE, shuffle=False, num_workers=8)

model = LightWrapper(model=UNet,
                     model_settings=dict(num_classes=1,
                                         in_channels=1,
                                         depth=4,
                                         start_filts=32,
                                         up_mode='transpose',
                                         merge_mode='concat'),
                     opt=AdamW,
                     opt_settings=dict(lr=1e-3),
                     loss=CrossEntropyLoss,
                     loss_settings=dict(),
                     scheduler=None,
                     scheduler_settings=dict())

trainer.fit(model, train_loader, test_loader)
logger.finalize()
