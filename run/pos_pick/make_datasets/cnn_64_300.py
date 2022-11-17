import sys
sys.path.append('/home/seleznev.ns/gradwork_ordered/')

from seispick.readsgy import Data
from seispick.pos_pick.utils.criteria import criteria_for_cut
from seispick.pos_pick.savedata import SaveData
from seispick.pos_pick.utils.compose import FunctionWrapperDouble, ComposeDouble

DATA_PATH = '/home/seleznev.ns/gradwork_ordered/run/pos_pick/raw_data/2564_1807_1819_1831_coord_H.sgy'
PATH_TO = '/home/seleznev.ns/gradwork_ordered/run/pos_pick/transformed_data/pospick_300/'

# PATH_TO = '../test/transformed_data/pospick_300/'

data = Data(path=DATA_PATH)

transforms = ComposeDouble([
                            FunctionWrapperDouble(
                                criteria_for_cut,
                                input=True, target=True,
                                window_size=300, settings=None, error_ratio=2),
                            ])

save_data = SaveData(data, transforms=transforms)
save_data.to_folder(PATH_TO)
