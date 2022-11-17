import sys
sys.path.append('/home/seleznev.ns/gradwork_ordered')

import torch
from seispick.pos_pick.fields import GetFields, WrapPositionPickingDataSet
from seispick.pos_pick.models.CNN import ConvNet_64_96
from seispick.pos_pick.datasets.transforms import (
    CustomCompose,
    CustomRandCrop,
    CustomCropByTargets,
    CustomChangeTargets,
    CustomAddDim,
    CustomToTensor,
    CustomToFloat)

# TEST_PATH = '../transformed_data/pospick_64/train'
TEST_PATH = './transformed_data/pospick_96/test'
SAVE_TO = './results'

V = 1442
WIDTH = 96

test_transforms_first = CustomCompose([CustomRandCrop(width=WIDTH)])

test_transforms_second = CustomCompose([CustomCropByTargets(V=V, width=64),
                                        CustomChangeTargets(),
                                        CustomAddDim(),
                                        CustomToTensor(),
                                        CustomToFloat(),
                                      ])

test_dataset = WrapPositionPickingDataSet(TEST_PATH, transforms=test_transforms_first)

checkpoint = torch.load('../ckpt/ckpt_ConvNet_64_96_(!1819)')
model = ConvNet_64_96()
model.load_state_dict(checkpoint['model_state_dict'])

test_dataset = GetFields(model=model,
                         data_set=test_dataset,
                         x_limits=40,
                         y_limits=40,
                         delta=1,
                         with_pics=True,
                         transforms=test_transforms_second,
                         batch_size=4,
                         num_workers=4)

test_dataset.save_fields(SAVE_TO)
print(test_dataset.all)
