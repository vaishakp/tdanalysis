" Carry out time-domain ringdown analysis using bilby "

import numpy as np
from TimeDomainAnalysis import CovarianceCalculator
from nr_injection import NRInjection
from os import mkdir
from os.path import isdir
from gwpy.timeseries import TimeSeries
import qnm
from nr_injection import MTSUN_SI

class RingdownInference:

    def __init__(self,
                    prior_ranges,
                    target_mass,
                    search_duration,
                    nr_data,
                    nlive=500,
                    dlogz=None,
                    sample_in_mass_spin=False,
                    analysis_duration=1,
                    correlation_duration=4,
                    ref_gps_time=None,
                    sampling_frequency=4096,
                    n_cov=1000,
                    noise_sampling_range=100,
                    gps_time_exclusion_duration=1,
                    auto_corr_method="median",
                    detector='L1',
                    t_start=0,
                    source_mass=1,
                    out_dir='./',
                    truths=[1e-19, 100, 0.5, 0]
                    ):

        # Covariance parameters
        self._CovarianceCalculator = CovarianceCalculator(ref_gps_time=ref_gps_time,
                                                          sampling_frequency=sampling_frequency,
                                                          correlation_duration=correlation_duration,
                                                          n_cov=n_cov,
                                                          noise_sampling_range=noise_sampling_range,
                                                          gps_time_exclusion_duration=gps_time_exclusion_duration,
                                                          auto_corr_method=auto_corr_method,
                                                          detector=detector,
                                                          analysis_duration=analysis_duration)



        # Injection parameters
        self._NRInection = NRInjection(target_mass=target_mass,
                                       analysis_duration=analysis_duration,
                                       search_duration=search_duration,
                                       nr_data=nr_data,
                                       t_start=t_start,
                                       target_sampling_frequency=sampling_frequency,
                                       source_mass=source_mass)


        # Inference parameters
        self._prior_ranges = prior_ranges
        self._nlive = nlive
        self._dlogz = dlogz
        self._sample_in_mass_spin=sample_in_mass_spin
        self._detector = detector
        #self._search_duration = search_duration

        self._truths = truths

        self._A0 = self.truths[0]


        self._time_axis = t_start + np.linspace(0, search_duration, search_duration*sampling_frequency)


    @property
    def prior_ranges(self):
        return self._prior_ranges
    
    @property
    def nlive(self):
        return self._nlive
    
    @property
    def dlogz(self):
        return self._dlogz
    
    @property
    def sample_in_mass_spin(self):
        return self._sample_in_mass_spin
    
    @property
    def detector(self):
        return self._detector
    

    @property
    def CovarianceCalculator(self):
        return self._CovarianceCalculator
    
    @property
    def NRInjection(self):
        return self._NRInection
    
    @property
    def out_dir(self):
        return self._out_dir
    
    @property
    def entire_data(self):
        return self._entire_data
    
    @property
    def Lij(self):
        return self._Lij
    
    @property
    def inv_Lij(self):
        return self._inv_Lij
    
    @property
    def search_data(self):
        return self._search_data
    
    @property
    def data(self):
        return self._data
    
    @property
    def noise_segment(self):
        return self._noise_segment
    
    @property
    def signal_segment(self):
        return self._signal_segment

    @property
    def data_segment(self):
        return self._data_segment

    @property
    def SNR(self):
        return self._SNR

    @property
    def time_axis(self):
        return self._time_axis
    
    @property
    def truths(self):
        return self._truths
    
    def initialize(self):

        if not isdir(self.out_dir):
            mkdir(self.out_dir)


        qnm.download_data()

        self.grav_220 = qnm.modes_cache(s=-2,l=2,m=2,n=0)
        #omega_inj, _, _ = grav_220(a=0.68)
        
        #print(omega_inj)


    def run(self):

        self.CovarianceCalculator.run()

        self.NRInjection.run()
        
        self.run


    def plot_entire_data(self):

        self.CovarianceCalculator.entire_noise_ts.plot()

    def cholesky_decompose(self):

        self._Lij = np.linalg.cholesky(self.CovarianceCalculator.Cij)

        self._inv_Lij = np.linalg.inv(self.Lij)

    def whiten(self, x):

        return np.dot(self.inv_Lij, x)

    def inner_product(self, x, y):

        xbar = self.whiten(x)
        ybar = self.whiten(y)

        return np.dot(xbar, ybar)  

    def assign_noise_segment(self):

        search_start = int( self.NRInjection.t_start*self.NRInjection.target_sampling_frequency )
        search_end = int( (self.NRInjection.t_start + self.NRInjection.search_duration)*self.NRInjection.target_sampling_frequency )

        self._noise_segment = self.CovarianceCalculator.entire_noise_ts[search_start:search_end]

        #self._data = self.search_data + self.A0*self.NRInjection.signal

    def assign_signal_segment(self):

        search_start = int( self.NRInjection.t_start*self.NRInjection.target_sampling_frequency )
        search_end = int( (self.NRInjection.t_start + self.NRInjection.search_duration)*self.NRInjection.target_sampling_frequency )

        self._signal_segment = self.A0*self.NRInjection.signal[search_start:search_end]

    def assign_data_segment(self):

        self._data_segment = self.noise_segment + self.signal_segment

    def compute_SNR(self):

        signal_seg_norm = self.inner_product(self.signal_segment, self.signal_segment)

        data_sig_inner_prod = self.inner_product(self.data_segment, self.signal_segment)

        mf_snr = data_sig_inner_prod/np.sqrt(signal_seg_norm)

        self._SNR = mf_snr

        print('Matched filter SNR', mf_snr)

    def plot_injection(self):

        import matplotlib.pyplot as plt

        plt.plot(self.time_axis, self.signal_segment)
        #plt.xlim(t_start, t_start+ans_dur)
        #plt.ylim(-2e-21, 2e-21)
        plt.show()


        plt.plot(self.time_axis, self.noise_segment)
        #plt.xlim(t_start, t_start+ans_dur)
        #plt.ylim(-2e-21, 2e-21)
        plt.show()

        plt.plot(self.time_axis, self.data_segment)
        #plt.xlim(t_start, t_start+(ans_dur))
        #plt.ylim(-2e-21, 2e-21)
        plt.show()


        plt.semilogy(self.time_axis, abs(self.data_segment))
        #plt.xlim(t_start, t_start+(ans_dur))
        #plt.ylim(-2e-21, 2e-21)
        plt.show()


    def time_domain_liklihood_direct(self, dx):

        dA, domega, dgamma, dphi = dx
        A0, omega0, gamma0, phi0 = self.truths

        dt = 0
        
        A = A0 +dA
        omega = omega0 + domega
        gamma = gamma0+dgamma
        t = self.NRInjection.t_start + dt
        phi = phi0 + dphi

        tloc = np.argmin(abs(self.time_axis -t))
        tloc_end = tloc + int(self.NRInjection.analysis_duration * self.NRInjection.target_sampling_frequency)

        data = self.data_segment[tloc:tloc_end]
        cdata = data - np.mean(data)

        local_time_axis = self.time_axis[tloc:tloc_end]
        signal =  A*np.cos(omega*(local_time_axis-t) + phi) * np.exp(-gamma*(local_time_axis-t))

        delta = cdata-signal

        # delta_bar = whiten(delta)

        LnP = (-1/2) * (self.inner_product(delta, delta))

        return LnP
    
    def time_domain_liklihood_mass_spin(self, dx):

        
        pars2 = self.transform_to_freq_tau(dx)

        return self.time_domain_liklihood_direct(dx)

    def transform_to_freq_tau(self):

        dA, dmass, dspin, dphi = dx

        
    def prior_transform_direct(self, u):

        x = np.array(u)


        x[0] = -50*self.A0 + u[0]*100*self.A0

        x[1] = max(0, self.truths[1] - 100) + u[1]*self.truths[1]*200

        x[2] =  max(0, self.truths[2] - 100) + u[2]*self.truths[2]*200

        x[3] = -np.pi/2 + u[3]*2*np.pi/2
        
        return x

    def prior_transform_mass_spin(self, u):

        x = np.array(u)


        x[0] = -50*self.A0 + u[0]*100*self.A0

        x[1] = max(0, self.truths[1] - 100) + u[1]*self.truths[1]*200

        x[2] =  max(0, self.truths[2] - 100) + u[2]*self.truths[2]*200

        x[3] = -np.pi/2 + u[3]*2*np.pi/2
        
        return x
    
    def signal_generator_direct(self, time_axis, pars):

        A, omega, gamma, phi = pars


        return A*np.cos(omega*(time_axis) + phi) * np.exp(-gamma*(time_axis))

    def signal_generator_mass_spin(self, time_axis, pars):

        A, omega, gamma, phi = pars


        return A*np.cos(omega*(time_axis) + phi) * np.exp(-gamma*(time_axis))


    def get_qnm_pars_from_mass_spin(self, pars):

        mass, spin = pars

        omega_inj, _, _ = self.grav_220(a=spin)

        omega_SI = omega_inj/(mass*MTSUN_SI)

        omega_re = omega_SI.real
        gamma = -omega_SI.imag

        freq = omega_re/(2*np.pi)
        tau = 1/gamma

        return freq, tau

    def run_sampler(self):