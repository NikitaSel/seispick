import sys
sys.path.append('/home/seleznev.ns/gradwork_ordered')

import torch
import albumentations as A
from seispick.time_pick.models.Unet import UNet
from seispick.time_pick.datasets.transforms import (
	FlattenTargets,
	InputsToMatrix,
	ToTensor,
	ToFloat,
)

from seispick.time_pick.datasets.dataset import ResultDataset
from seispick.time_pick.save_results import SaveResults


TEST_PATH = '/home/seleznev.ns/gradwork_ordered/run/time_pick/transformed_data/test/data'
TEST_PATH = './transformed_data/test/data'
SAVE_TO = './results'

V = 1442
BATCH_SIZE = 20

test_transforms = A.Compose([FlattenTargets(epsilon=0.01),
							  InputsToMatrix(size=32),
							  ToTensor(),
							  ToFloat(),
                            ])

test_data = ResultDataset(TEST_PATH, transforms=test_transforms)

checkpoint = torch.load('../ckpt/ckpt_UNet_(!1819)')
model = UNet(num_classes=1,
             in_channels=1,
             depth=4,
             start_filts=32,
             up_mode='transpose',
             merge_mode='concat')
model.load_state_dict(checkpoint['model_state_dict'])

res = SaveResults(path_to=SAVE_TO,
                     dataset=test_data,
                     model=model,
                     save_img=True)
res.save()
print(res.score)