from os import listdir
from sys import prefix
import numpy as np
from gwpy.timeseries import TimeSeries
from waveformtools.waveformtools import message, progressbar, roll
import os
from os.path import isfile
import pickle
from pathlib import Path
import dask


class CovarianceCalculator:
    """Carry out time-domain analysis of data.

    Given a gps reference time, this allows

    1.  Computation of the autocorrelation function of specified length.
        This will be computed as either the median or mean over `n_cov`
        auto-correlation frames.
    2.  Construction of the covariance matrix from the autocorrelation function
        assuming a symmetric Toeplez form.
    3.  Inference over the given priors `priors` and `n_live` number of live
        points with a specified model.


    """

    def __init__(
        self,
        ref_gps_time,
        sampling_frequency=4096,
        correlation_duration=4,
        n_cov=10,
        noise_sampling_range=100,
        gps_time_exclusion_duration=8,
        auto_corr_method="median",
        detector="L1",
        channel="GDS-CALIB_STRAIN_CLEAN",
        analysis_duration=1,
        data_type="open",
        backend="dask",
        chunks=10000,
    ):
        self._sampling_frequency = sampling_frequency
        self._correlation_duration = correlation_duration
        self._n_cov = n_cov
        self._noise_sampling_range = noise_sampling_range
        self._ref_gps_time = ref_gps_time
        self._gps_time_exclusion_duration = gps_time_exclusion_duration
        self._auto_corr_method = auto_corr_method
        self._detector = detector
        self.channel = channel
        self.data_type = data_type
        self.backend = backend
        self.chunks = chunks

        if self.data_type == "open":
            self.load_channel = self.detector
            self.get_data = TimeSeries.fetch_open_data

        elif self.data_type == "server":
            self.load_channel = f"{self.detector}:{self.channel}"
            self.get_data = TimeSeries.get

        else:
            self.load_channel = None

        self._all_moments = {}

        self._N = self.sampling_frequency * self.correlation_duration
        self._analysis_duration = analysis_duration

        self._N_analysis = int(self.sampling_frequency * self.analysis_duration)

        self._nothing_todo = False
        self._compute_Cij = True
        self._download = True
        self._compute_moments = True

        self.setup_backend()

        # self.initialize()

        if self.auto_corr_method == "mean":
            self.estimator = np.mean
        elif self.auto_corr_method == "median":
            self.estimator = np.median
        else:
            raise KeyError(f"Unknown estimator {self.auto_corr_method}")

    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    @property
    def correlation_duration(self):
        return self._correlation_duration

    @property
    def n_cov(self):
        return self._n_cov

    @property
    def noise_sampling_range(self):
        return self._noise_sampling_range

    @property
    def ref_gps_time(self):
        return self._ref_gps_time

    @property
    def noise_segments(self):
        return self._noise_segments

    @property
    def N(self):
        return self._N

    @property
    def N_analysis(self):
        return self._N_analysis

    @property
    def est_auto_corr(self):
        return self._est_auto_corr

    @property
    def gps_time_exclusion_duration(self):
        """The duration of the time segment to exclude
        from analysis. This is specified as a single
        integer number. The segment is assumed to be
        centered at `gps_time`"""

        return self._gps_time_exclusion_duration

    @property
    def auto_corr_method(self):
        return self._auto_corr_method

    @property
    def detector(self):
        return self._detector

    @property
    def entire_segment_gps_start_time(self):
        return self._entire_segment_gps_start_time

    @property
    def entire_segment_gps_end_time(self):
        return self._entire_segment_gps_end_time

    @property
    def entire_noise_ts(self):
        return self._entire_noise_ts

    @property
    def data_file_name(self):
        return self._data_file_name

    @property
    def all_auto_corr(self):
        return self._all_auto_corr

    @property
    def Cij(self):
        return self._Cij

    @property
    def Cij_file_name(self):
        return self._Cij_file_name

    @property
    def all_moments(self):
        return self._all_moments

    @property
    def analysis_duration(self):
        return self._analysis_duration

    @property
    def nothing_todo(self):
        return self._nothing_todo

    @property
    def moments_file_name(self):
        return self._moments_file_name

    @property
    def compute_Cij(self):
        return self._compute_Cij

    @property
    def download(self):
        return self._download

    @property
    def compute_moments(self):
        return self._compute_moments

    def setup_backend(self):

        if self.backend == "numpy":
            from numpy import mean, median, sqrt, dot

            self.zeros = np.zeros
            self.array = np.array

            self.mean = np.mean
            self.median = np.median
            self.sqrt = np.sqrt
            self.dot = np.dot

        elif self.backend == "dask":
            from functools import partial
            import dask.array

            self.zeros = partial(dask.array.wrap.zeros, chunks=self.chunks)
            self.array = partial(dask.array.from_array, chunks=self.chunks)

            self.mean = dask.array.reductions.mean
            self.median = dask.array.reductions.median
            self.sqrt = dask.array.ufunc.sqrt
            self.dot = dask.array.routines.dot

        else:
            raise KeyError(f"Unknown backend {self.backend}")

    def initialize(self):

        message("Initializing")
        message("\tParsing GPS times...")

        self._Cij_file_name = f"Cij_gpsT{self.ref_gps_time}_R{self.noise_sampling_range}_D{self.correlation_duration}_S{self.sampling_frequency}_A{self.analysis_duration}_N{self.n_cov}"
        self._moments_file_name = f"moments_gpsT{self.ref_gps_time}_R{self.noise_sampling_range}_D{self.correlation_duration}_S{self.sampling_frequency}_A{self.analysis_duration}_N{self.n_cov}.pkl"

        if self.data_type == "open" or self.data_type == "server":
            self.setup_online_data()

        else:
            # self.find_data_segment()
            # self.load_data_propreitery()
            # self._data_file_name = f"Entire_noise_ts_{self.data_type}_gpsTref{self.ref_gps_time}_{self.entire_segment_gps_start_time}_{self._entire_segment_gps_end_time}_S{self.sampling_frequency}_A{self.analysis_duration}.txt"
            raise NotImplementedError

    def setup_online_data(self):
        """Download, and read in open data"""
        self._entire_segment_gps_start_time = (
            self.ref_gps_time - self.noise_sampling_range - self.correlation_duration
        )
        self._entire_segment_gps_end_time = (
            self.ref_gps_time + self.noise_sampling_range + self.correlation_duration
        )
        message("\tSetting file names...")
        self._data_file_name = f"Entire_noise_ts_{self.data_type}_gpsT{self.ref_gps_time}_R{self.noise_sampling_range}_D{self.correlation_duration}_S{self.sampling_frequency}_A{self.analysis_duration}.txt"

        message("Obtaining data")
        if self.check_file_exists(self.data_file_name):
            message("\tLoading data from disk", message_verbosity=1)
            self.load_entire_time_segment()

            self._download = False

            if self.check_file_exists(self.Cij_file_name):
                message("Loading Cij from disk", message_verbosity=1)
                self.load_Cij()

                self._compute_Cij = False

                if self.check_file_exists(self.moments_file_name):
                    self.load_stat_moments()

                    self._compute_moments = False

                    self._nothing_todo = True

        else:
            message("\tDownloading data")
            self.download_full_data()

            self.resample_data()
            message("\t Saving data to disk")
            self.save_entire_time_segment()

    def resample_data(self):

        original_sampling_f = self.entire_noise_ts.sample_rate

        if int(self.sampling_frequency) != int(original_sampling_f):
            message("Resampling using cubic spline", message_verbosity=2)

            from scipy.interpolate import interp1d

            new_dt = 1 / self.sampling_frequency
            new_time_axis = np.arange(
                self.entire_noise_ts_time_axis[0],
                self.entire_noise_ts_time_axis[-1],
                new_dt,
            )
            new_entire_noise_ts = interp1d(
                self.entire_noise_ts_time_axis, self.entire_noise_ts, kind="Cubic"
            )(new_time_axis)

            self._entire_noise_ts = TimeSeries(
                data=new_entire_noise_ts, times=new_time_axis
            )
            self.entire_noise_ts_time_axis = new_time_axis

    def compute_effective_noise_segment_gps_times(self):
        """Get the noise segment GPS time endpoints of correlation duration
        that do not include the gps_time_exclusion duration.
        Returns a dict.
        """

        message("Computing noise segment GPS times")
        noise_segments = {}
        for seg_ind in range(self.n_cov):
            not_found = True

            while not_found:
                seg_centre = np.random.randint(
                    self.ref_gps_time - self.noise_sampling_range,
                    self.ref_gps_time + self.noise_sampling_range,
                )
                if not (
                    self.ref_gps_time - int(self.gps_time_exclusion_duration / 2)
                    <= seg_centre
                    <= (self.ref_gps_time + int(self.gps_time_exclusion_duration / 2))
                ):
                    not_found = False

            noise_segments.update(
                {
                    seg_ind: (
                        seg_centre - int(self.correlation_duration / 2),
                        seg_centre + int(self.correlation_duration / 2),
                    )
                }
            )

        self._noise_segments = noise_segments

    def get_segment_gps_times(self, segment_number):

        return self.noise_segments[segment_number]

    def fetch_data_array(self, segment_number):

        if self.check_array_exists(self.entire_noise_ts):
            message("Fetching saved data", message_verbosity=3)
            # print(self.get_segment_gps_times(segment_number))
            seg_gsp_start, seg_gps_end = self.get_segment_gps_times(segment_number)
            ind_start = (
                seg_gsp_start - self.entire_segment_gps_start_time
            ) * self.sampling_frequency
            ind_end = (
                seg_gps_end - self.entire_segment_gps_start_time
            ) * self.sampling_frequency
            noise_ts = self.entire_noise_ts[ind_start:ind_end]

        else:
            try:
                noise_ts = self.get_data(
                    self.channel,
                    *self.get_segment_gps_times(segment_number),
                    verbose=True,
                )

            except Exception as excep:
                message(excep)

        return self.demean_ts(noise_ts)

    def check_array_exists(self, array):

        # if self.backend=='numpy':
        return not (np.array(self.entire_noise_ts) == np.array(None)).all()
        # elif self.backend=='dask':
        #    return not (self.entire_noise_ts == None).all().compute()
        # else:
        #    raise KeyError(f"Unknown backend {self.backend}")

    def estimate_auto_corr(self):
        """Estimate the auto-correaltion function using all
        the noise segments"""

        message("Computing autocorrelations...\n", message_verbosity=1)

        all_auto_corr = np.zeros((self.n_cov, self.N_analysis))

        for seg_ind in range(self.n_cov):

            noise_ts = np.array(self.fetch_data_array(seg_ind))
            one_set_of_moments = self.compute_stat_moments(noise_ts)
            self._all_moments.update({seg_ind: one_set_of_moments})
            auto_corr_i = self.auto_coorelate(noise_ts)
            all_auto_corr[seg_ind, :] = auto_corr_i
            progressbar(seg_ind + 1, self.n_cov)

        self._all_auto_corr = all_auto_corr

        est_auto_corr = np.array(
            [self.estimator(all_auto_corr[:, ind]) for ind in range(self.N_analysis)]
        )

        self._est_auto_corr = est_auto_corr
        message("\n", message_verbosity=1)

    def estimate_symm_toeplez_Cij(self):
        """Construct a symmetric Toeplex cross correlation
        matrix from a given a noise time series"""

        message("\nComputing the cross correlation matrix")

        auto_corr = self.est_auto_corr

        N = self.N_analysis

        Cij = self.zeros((N, N))

        count = 0
        total_num_entries = N * (N + 1) / 2

        if self.backend == "numpy":
            row_idx, col_idx = np.indices(Cij.shape)
        elif self.backend == "dask":
            row_idx, col_idx = dask.array.indices(Cij.shape)
        else:
            raise KeyError(f"Unknown backend {self.backend}")

        for rel_idx, acor_val in enumerate(auto_corr):
            Cij[abs(row_idx - col_idx) == rel_idx] = acor_val
            count += 1
            print(f"Progress: {count/total_num_entries}%", end="\r")

        # for row_ind in range(N):
        #    for col_ind in range(row_ind, N):

        #        rel_ind = abs(col_ind - row_ind)
        #        Cij[row_ind, col_ind] = auto_corr[rel_ind]

        #        if row_ind != col_ind:
        #            Cij[col_ind, row_ind] = Cij[row_ind, col_ind]

        #        count += 1

        #        print(f"Progress: {count/total_num_entries}%", end='\r')

        # if count % 1000 == 0:
        #    progressbar(count, total_num_entries)

        self._Cij = Cij

    def run(self):

        if not self.nothing_todo:
            self.compute_effective_noise_segment_gps_times()

            if self.compute_moments:
                self.estimate_auto_corr()
                self.save_stat_moments()

            if self.compute_Cij:
                self.estimate_symm_toeplez_Cij()
                self.save_Cij()

    def download_full_data(self):
        """Fetch the entire noise stream for the required duration"""

        message("Downloading data...", message_verbosity=1)

        gps0 = self.ref_gps_time - self.noise_sampling_range - self.correlation_duration
        gps1 = self.ref_gps_time + self.noise_sampling_range + self.correlation_duration

        try:
            entire_noise_ts = self.get_data(self.load_channel, gps0, gps1, verbose=True)

        except Exception as excep:
            message(excep)
            entire_noise_ts = None

        entire_noise_ts = self.demean_ts(entire_noise_ts)

        self.entire_noise_ts_time_axis = entire_noise_ts.times
        self._entire_noise_ts = entire_noise_ts

    def demean_ts(self, time_series):
        # if isinstance(time_series, TimeSeries):
        if self.backend == "numpy" or isinstance(time_series, np.ndarray):
            return time_series - np.mean(time_series)
        # elif isinstance(self, da.array):
        elif self.backend == "dask" and isinstance(time_series, da.array):
            return time_series - self.mean(time_series).compute()

    def find_data_segment(self):
        dir_suffix = int(self.ref_gps_time / 10000000)

        assert (
            dir_suffix > 99 and dir_suffix < 1000
        ), f"The prefix dir could not be computed properly from the reference time {dir_suffix}"

        self.data_file_dir = Path(
            f"/cvmfs/ligo.storage.igwn.org/igwn/ligo/frames/O4/hoft_C00/{self.detector}/{self.detector[0]}-{self.detector}_HOFT_C00-{dir_suffix}"
        )
        self._data_file_name = self.get_dataframe_name_containing_reference_time(
            self.data_file_dir
        )
        self.data_file_path = self.data_file_dir / self.data_file_name

    def load_data_propreitery(self):
        """Download propreitery LIGO data from cvmfs.

        Find the segment that contains the trigger time
         Usually aggregated data a large. It should be enough
          for PSD estimation"""

        entire_ts = TimeSeries.read(
            self.data_file_path, f"{self.detector}:{self.channel}"
        )

        entire_ts = self.demean_ts(entire_ts)

        self.entire_noise_ts_time_axis = self.entire_noise_ts.times

        self._entire_noise_ts = entire_ts

        self._entire_segment_gps_start_time = self.entire_noise_ts_time_axis[0]
        self._entire_segment_gps_end_time = self.entire_noise_ts_time_axis[-1]

    def get_dataframe_name_containing_reference_time(self, data_file_dir):
        """Get available gps times"""

        files = os.listdir(data_file_dir)

        files = [item for item in files if ".gwf" in item]

        self.available_gps_times = np.array([int(item[14:24]) for item in files])

        data_file_idx = np.argmin(abs(self.available_gps_times - self.ref_gps_time))
        data_file_name = files[data_file_idx]

        return data_file_name

    def save_entire_time_segment(self):

        entire_noise_ts = TimeSeries(
            data=self.entire_noise_ts, times=self.entire_noise_ts_time_axis
        )
        entire_noise_ts.write(self.data_file_name)

    def load_entire_time_segment(self):
        entire_noise_ts = TimeSeries.read(self.data_file_name)

        self.entire_noise_ts_time_axis = entire_noise_ts.times
        self._entire_noise_ts = entire_noise_ts

    def check_file_exists(self, file_path):
        flag = isfile(file_path)
        return flag

    def auto_coorelate(self, ts):
        """Autocorrelate a astretch of data"""

        autocorr = []

        for index in range(0, self.N_analysis):

            # print(index/N)
            ts2 = roll(np.array(ts), index)

            autocorr.append(np.dot(np.array(ts), np.array(ts2)) / self.N)

        return autocorr

    def save_Cij(self):
        """Save the covariance matrix"""

        message("Saving the covariance matrix", message_verbosity=2)

        if isfile(self.Cij_file_name):
            raise FileExistsError(f"{self.Cij_file_name} File already exists!")

        else:
            self.save(self.Cij_file_name, self.Cij)

    def load_Cij(self):
        if isinstance(np.ndarray, array):
            self._Cij = np.load(f"{self.Cij_file_name}.npy")
        elif isinstance(self, da.array):

            f = h5py.File(f"{self.Cij_file_name}.h5")["/Cij"]
            self._Cij = dask.array.from_array(f)
            f.close()

    def save_array(self, file_name, array):

        if isinstance(np.ndarray, array):
            np.save(f"{file_name}.npy", array)

        elif isinstance(self, da.array):
            return array.to_hdf5(f"{file_name}.h5", "/Cij")

    def compute_stat_moments(self, noise_ts):

        noise_array = np.array(noise_ts)

        # Mean
        noise_array_mean = np.mean(noise_array)

        # Variance
        noise_array_variance = np.mean((noise_array - noise_array_mean) ** 2)
        noise_array_std = self.sqrt(noise_array_variance)

        moments = [noise_array_mean, noise_array_variance]
        # Higher moments
        for mindex in range(3, 13):

            df = self.double_factorial(mindex - 1)

            moments.append(
                np.mean(((noise_array - noise_array_mean) / noise_array_std) ** mindex)
                / df
            )

        return moments

    def save_stat_moments(self):

        with open(self.moments_file_name, "wb") as f:
            pickle.dump(self.all_moments, f)

    def load_stat_moments(self):

        with open(self.moments_file_name, "rb") as f:
            self._all_moments = pickle.load(f)

    def get_nth_moment(self, n):

        nth_moment = [self.all_moments[ind][n] for ind in self.all_moments.keys()]

        return nth_moment

    def plot_moments(self):

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(18, 12))

        for mxind in range(4):
            for myind in range(3):

                mn = mxind * 3 + myind

                ax[mxind, myind].scatter(
                    self.all_moments.keys(), self.get_nth_moment(mn - 1)
                )
                ax[mxind, myind].set_title(f"Moment {mn+1}")

        plt.show()

    def double_factorial(self, n):

        if n == 0 or n == 1:
            return 1

        else:
            return n * self.double_factorial(n - 2)
