import sys

sys.path.append("/home/vaishakprasad/Projects/custom_libraries/tdanalysis")
from tdanalysis.TimeDomainAnalysis import CovarianceCalculator
import matplotlib.pyplot as plt
import config
import numpy as np

config.conf_matplolib()

from config.verbosity import levels

vlev = levels()
vlev.set_print_verbosity(1)

tdal = CovarianceCalculator(
    ref_gps_time=1420878140,
    noise_sampling_range=512,
    correlation_duration=64,
    n_cov=128,
    gps_time_exclusion_duration=16,
    analysis_duration=14,
    data_type="server",
    backend="dask",
    chunks=100000,
    sampling_frequency=4096,
)


tdal.initialize()

tdal.run()


plt.plot(tdal.est_auto_corr)
plt.savefig("est_acorr.pdf")
