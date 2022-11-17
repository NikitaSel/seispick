import segyio
import numpy as np


class BaseIterible1:
    def __init__(self, elements, ind2key):
        self.elements = elements
        self._ind2key = ind2key
        
        if self._ind2key is None: 
            self._ind2key = {ind: key for ind, key in enumerate(self.elements)}
    
    def __iter__(self):
        self._i = 0
        return self
    
    def __next__(self):
        if self._i < len(self):
            self._i += 1
            return self.elements[self._ind2key[self._i - 1]]
        else:
            raise StopIteration
            
    def __len__(self):
        return len(self.elements)


class BaseIterible2:
    def __init__(self):
        pass
    
    def __iter__(self):
        self._i = 0
        self._it = None
        return self
    
    def __next__(self):
        try:
            if self._it is None:
                raise ValueError
            return next(self._it)
        except (StopIteration, ValueError):
            self._i += 1

            if self._i == len(self) + 1:
                raise StopIteration

            self._it = iter(self.elements[self._ind2key[self._i - 1]])

            return next(self._it)
        
    def __len__(self):
        return len(self.elements)


class BaseData:
    def __init__(self, data, cls, info):
        self.elements = {}
        self.info = info if info else dict()
        self._circum = {'0': None, '1': None}
        
        self.__circum(cls, data)
        self.__signal_shape(data)
        
        for elem in np.unique(np.sort(self._circum['0'])): ## если нумерация совпадает со временем, иначе сортировать по времени
            mask = (self._circum['0'] == elem)
            tmp_data = {key: data[key][mask] if key != 'traces' else data[key][mask, :] for key in data}
            
            info = self.info.copy()
            info[self._info_helper(cls)] = elem
            
            self.elements[elem] = self._circum['1'](tmp_data, info)

    def _info_helper(self, cls):
        cls_name2key = {'Data': 'Station', 'Station': 'Line', 'Line': 'Point'}
        return cls_name2key[cls.__class__.__name__]
            
    def __circum(self, cls, data):
        if isinstance(cls, Station):
            self._circum['0'] =  data['sp']
            self._circum['1'] =  Line
            
        elif isinstance(cls, Line):
            self._circum['0'] =  data['tr_seq']
            self._circum['1'] =  Point
        elif isinstance(cls, Data):
            self._circum['0'] =  data['cdp']
            self._circum['1'] =  Station
        else:
            assert False, "wrong cls"
            
    def __getitem__(self, key):
        return self.elements[key]
    
    def __signal_shape(self, data):
        self.signal_shape = data['traces'].shape
    
    def __len__(self):
        return len(self.elements)
    
    def ind2key(self):
        return {ind: key for ind, key in enumerate(self.elements.keys())}
    
    def stack_points(self):
        ret_val = {
                   'traces': np.zeros(shape=self.signal_shape),
                   'gx': np.zeros(shape=self.signal_shape[0]),
                   'gy': np.zeros(shape=self.signal_shape[0]),
                   'sx': np.zeros(shape=self.signal_shape[0]),
                   'sy': np.zeros(shape=self.signal_shape[0]),
                   'rge': np.zeros(shape=self.signal_shape[0]),
                   'sd': np.zeros(shape=self.signal_shape[0]),
                   'tr_interval': np.zeros(shape=self.signal_shape[0]),
                   'drt': np.zeros(shape=self.signal_shape[0])
                  }
        
        points = iter(self)
        
        for i, point in enumerate(points):
            for key in ret_val.keys():
                ret_val[key][i] = point.data[key][0]
        return ret_val


class Data(BaseData, BaseIterible2):
    def __init__(self, path):
        self.seg_path = path
        
        with segyio.open(self.seg_path, ignore_geometry=True) as f:
            coord_scaler = np.array(f.attributes(segyio.TraceField.SourceGroupScalar)[:])
            data = {
                     "cdp": np.array(f.attributes(segyio.TraceField.CDP)[:]),
                     "sp": np.array(f.attributes(segyio.TraceField.ShotPoint)[:]) // 10000 , #coord
                    #  "sp":np.array(f.attributes(segyio.TraceField.TRACE_SEQUENCE_FILE)[:]), #UK
                     "traces": np.array(f.trace.raw[:]),
                     "gx": np.array(f.attributes(segyio.TraceField.GroupX)[:] / coord_scaler), 
                     "gy": np.array(f.attributes(segyio.TraceField.GroupY)[:] / coord_scaler), 
                     "sx": np.array(f.attributes(segyio.TraceField.SourceX)[:] / coord_scaler), 
                     "sy": np.array(f.attributes(segyio.TraceField.SourceY)[:] / coord_scaler),
                     "rge": np.array(f.attributes(segyio.TraceField.ReceiverGroupElevation)[:] / coord_scaler),
                     "sd": np.array(f.attributes(segyio.TraceField.SourceDepth)[:] / coord_scaler),
                     "tr_interval": np.array(f.attributes(segyio.TraceField.TRACE_SAMPLE_INTERVAL)[:]),
                     "drt": np.array(f.attributes(segyio.TraceField.DelayRecordingTime)[:]),
                     "tr_seq": np.array(f.attributes(segyio.TraceField.TRACE_SEQUENCE_FILE)[:]) #coord
                    #  "tr_seq": np.array(f.attributes(segyio.TraceField.TRACE_SEQUENCE_LINE)[:]) #UK
                    }

            mask = data['cdp'] != 1819
            for key in data:
                data[key] = data[key][mask]

        f.close()
        
        super().__init__(data, self, None)
        self.stations = np.unique(self._circum['0'])
        self._ind2key = self.ind2key()


class Station(BaseData, BaseIterible2):
    def __init__(self, data, info):
        super().__init__(data, self, info)
        self.lines = np.unique(self._circum['0'])
        self._ind2key = self.ind2key()


class Line(BaseData, BaseIterible1):
    def __init__(self, data, info):
        super().__init__(data, self, info)
        self.points = np.unique(self._circum['0'])
        self._ind2key = self.ind2key()


class Point:
    def __init__(self, data, info):
        self.data = data
        self.info = info
        self.signal_shape = self.data['traces'].shape
