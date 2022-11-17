import sys
sys.path.append('/home/seleznev.ns/gradwork_ordered/')

from seispick.readsgy import Data
from seispick.time_pick.savedata import SaveSeiSignals

from seispick.time_pick.utils.criteria import criteria_for_cut
from seispick.time_pick.utils.utils import crop
from seispick.time_pick.utils.compose import FunctionWrapperDouble, ComposeDouble


DATA_PATH = '/home/seleznev.ns/gradwork_ordered/run/pos_pick/raw_data/2564_1807_1819_1831_coord_H.sgy'
PATH_TO = '/home/seleznev.ns/gradwork_ordered/run/time_pick/transformed_data/'

# PATH_TO = '/home/seleznev.ns/gradwork_ordered/run/time_pick/test/transformed_data/'

V = 1442

data = Data(path=DATA_PATH)

transforms = ComposeDouble([
                            FunctionWrapperDouble(criteria_for_cut,
                                                  input=True, target=True,
                                                  V=V),
                             FunctionWrapperDouble(crop,
                                                   input=True, target=True,
                                                   V=V, width=304, mess=True),
                            ])

save_data = SaveSeiSignals(PATH_TO, 
                           data, V=V, 
                           transforms=transforms, 
                           save_img=False)
save_data.save()
