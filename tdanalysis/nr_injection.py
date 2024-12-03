''' A class to inject a numerical relativity waveform 
into data '''

import numpy as np
from waveformtools.waveformtools import message
from copy import deepcopy

MTSUN_SI = 4.925490947641267e-06

class NRInjection:

    def __init__(self,
                 analysis_duration,
                 search_duration,
                 nr_data,
                 target_mass=None,
                 t_start=0,
                 target_sampling_frequency=4096,
                 source_mass=1,
                 ):
        
        self._source_mass = source_mass
        self._target_sampling_frequency = target_sampling_frequency
        self._target_mass = target_mass
        self._analysis_duration = analysis_duration
        self._search_duration = search_duration
        self._nr_data = nr_data
        self._t_start = t_start
    
        self._time_axis = nr_data[:, 0]

        self._frame = 'original'

        if self._target_mass is None:
            self.find_target_mass()
        


    @property
    def source_mass(self):
        return self._source_mass
    
    @property
    def t_start(self):
        return self._t_start
    
    @property
    def target_sampling_frequency(self):
        return self._target_sampling_frequency
    
    @property
    def target_mass(self):
        return self._target_mass
    
    @property
    def analysis_duration(self):
        return self._analysis_duration
    
    @property
    def search_duration(self):
        return self._search_duration
    
    @property
    def nr_data(self):
        return self._nr_data

    @property
    def time_axis(self):
        return self._time_axis
    
    @property
    def target_delta_t(self):
        return self._target_delta_t
    
    @property
    def original_delta_t(self):
        return self._orignial_delta_t

    @property
    def frame(self):
        return self._frame
    
    @property
    def delta_t(self):
        return self.time_axis[-1] - self.time_axis[-2]
    
    @property
    def transformed_nr_data(self):
        return self._transformed_nr_data
    
    @property
    def signal(self):
        return self._signal
    
    def transform_to_unit_frame(self):

        if "unit" not in self.frame:

            self._time_axis *= self.source_mass

            self._frame+="_unit"
        else:
            message("Already in Unit frame")

    def transform_to_solar_mass_frame(self):
        
        if 'unit' not in self.frame:
            self.transform_to_unit_frame()

        if "solarmass" not in self.frame:
            self._time_axis*= self.target_mass*MTSUN_SI
        else:
            message("Already in solarmass frame")

    def find_target_mass(self):
        ''' Find the target mass from the `target_sampling_frequency` '''

        if 'unit' not in self.frame:
            self.transform_to_unit_frame()

        if 'solarmass' in self.frame:
            raise ValueError("Data should not already be in Solar Mass frame.")
        
        Mtarget = self.target_delta_t/(self.delta_t * MTSUN_SI)

        self._target_mass = Mtarget

    
    def get_output(self):
        ''' Get the transformed data array '''

        transformed_nr_data = deepcopy(self.nr_data)
        transformed_nr_data[:, 0] = self.time_axis

        self._transformed_nr_data = transformed_nr_data

        return transformed_nr_data
    

    def check_signal_duration(self):

        if 'solarmass' in self.frame:

            data_duration = self.delta_t*len(self.time_axis)

            if data_duration!=self.analysis_duration:
                raise ValueError(f" The data duration is {data_duration} while the target duration is {self.analysis_duration}")

        else:
            raise KeyError("Transform to solarmass frame first!")
        
        
    def prepare_signal(self):
        ''' Prepare the signal to be injected into the search noise stream '''

        Ntotal = self.target_sampling_frequency*self.analysis_duration

        self._signal = np.zeros(Ntotal)

        start_ind = int(self.t_start*self.target_sampling_frequency)

        end_ind = int((self.t_start + self.analysis_duration)*self.target_sampling_frequency)

        cur = end_ind - start_ind

        if cur!= int(self.analysis_duration*self.target_sampling_frequency):
            raise ValueError("The computed length of signal is not consistent with the requested values")
        
        self._signal[start_ind:end_ind] = self.transformed_nr_data[:, 1]


    def run(self):

        self.transform_to_unit_frame()
        self.transform_to_solar_mass_frame()
        self.prepare_signal()