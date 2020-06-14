"""
set of utility functions for handling SuperWASP FITS data
"""

import matplotlib.pyplot as plt
import numpy as np

from astropy import units as u
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.timeseries import LombScargle
from astropy.time import Time
from astropy.timeseries import aggregate_downsample
from astropy.timeseries import TimeSeries

__all__ = ["lightcurve"]

TIMESTAMP_COLUMN = 'TMID'
FLUX_COLUMN = 'TAMFLUX2'
FLUX_ERR_COLUMN = 'TAMFLUX2_ERR'
MAG_COLUMN = 'mag'
MAG_ERR_COLUMN = 'mag_err'

def boundify(fluxes, sigma=2.0):
    mean = fluxes.mean()
    dev = np.std(fluxes)

    return (mean - (sigma * dev)).value, (mean + (sigma * dev)).value

class lightcurve:
    period = None
    ts = None

    def __init__(self, filename, normalize_phase=False, period=None):
        t = Table.read(filename, hdu=1)
        t[TIMESTAMP_COLUMN].unit = 'second'
        """ set timestamp to julian date """
        t['time'] = (t[TIMESTAMP_COLUMN] / 86400.0) + 2453005.5
        t['time'].unit = 'day'
        t['time'] = Time(t['time'], format='jd')

        t[FLUX_COLUMN].unit = 'mag'
        t[FLUX_ERR_COLUMN].unit = 'mag'

        """ calculate apparent visual magnitude """
        t[MAG_COLUMN] = 15.0 - (2.5 * np.log10(t[FLUX_COLUMN]))
        t[MAG_COLUMN].unit = 'mag'

        """ calculate magnitude error """
        t[MAG_ERR_COLUMN] = 1.08574 * t[FLUX_ERR_COLUMN]
        t[MAG_ERR_COLUMN].unit = 'mag'

        t.keep_columns(['time', TIMESTAMP_COLUMN, FLUX_COLUMN, FLUX_ERR_COLUMN, MAG_COLUMN, MAG_ERR_COLUMN])



        self.ts = TimeSeries(t)
        self.period = period

        if self.period is None:
            self.period = self.calculate_period()

        if normalize_phase:
            wrap_phase = 1.0
        else:
            wrap_phase = self.period

        self.ts = self.ts.fold(period=self.period, normalize_phase=False, wrap_phase=wrap_phase)
        self.ts.sort('time')

    @property
    def time(self):
        return self.ts.time

    @property
    def flux(self):
        return self.ts[FLUX_COLUMN]

    @property
    def flux_error(self):
        return self.ts[FLUX_ERR_COLUMN]

    @property
    def magnitude(self):
        return self.ts[MAG_COLUMN]
    
    @property
    def magnitude_error(self):
        return self.ts[MAG_ERR_COLUMN]

    def calculate_period(self):
        min_period = 2.0 * u.hour
        max_period = 12.0 * u.hour

        periodogram = LombScargle.from_timeseries(
            self.ts,
            signal_column_name=FLUX_COLUMN,
            uncertainty=FLUX_ERR_COLUMN,
            nterms=6
        )

        frequency, power = periodogram.autopower(
            nyquist_factor=5,
            minimum_frequency=1.0 / (max_period.to(u.day)),
            maximum_frequency=1.0 / (min_period.to(u.day)),
            samples_per_peak=10
        )

        best = power.argmax()
        
        return 1.0 / frequency[best]

    def normalize(self):
        mean, median, stddev = sigma_clipped_stats(self.ts[FLUX_COLUMN])
        self.ts[FLUX_COLUMN] = self.ts[FLUX_COLUMN] / median

    def write(self, filename, format=None, *args, **kwargs):
        return self.ts.write(filename, format=format, *args, **kwargs)

    def plot(self, show_mean=False):
        y_min, y_max = boundify(self.flux)
        fig = plt.figure(figsize=(20,20))

        ax1 = fig.add_subplot(111)

        # draw 2 phases of the base light-curve
        ax1.plot((self.time.jd / self.period.to(u.day)).value, self.flux, 'k.', markersize=1)
        ax1.plot((self.time.jd / self.period.to(u.day)).value - 1.0, self.flux, 'k.', markersize=1)

        if show_mean:
            lc_mean = aggregate_downsample(self.ts, time_bin_size=0.001 * u.day)
            # draw 2 phases of the mean 
            ax1.plot((lc_mean.time_bin_start.jd / self.period.to(u.day)).value, lc_mean[FLUX_COLUMN], 'r-', drawstyle='steps', markersize=1)
            ax1.plot((lc_mean.time_bin_start.jd / self.period.to(u.day)).value - 1.0, lc_mean[FLUX_COLUMN], 'r-', drawstyle='steps', markersize=1)

        ax1.set_ylim(bottom=y_min, top=y_max)
        ax1.set_xlabel('Phase')
        ax1.set_ylabel('Flux (micro vega)')

        return fig, ax1


    