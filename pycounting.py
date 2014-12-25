# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import curve_fit
from operator import itemgetter
import glob
import matplotlib.pyplot as plt
import cycounting as _cycounting
from itertools import product
import datetime
from textwrap import dedent
import h5py
import abc
from collections import defaultdict


def current_time(format='%Y/%m/%d %H:%M:%S'):
    return datetime.datetime.now().strftime(format)


def tc(sampling_rate, unit='s'):
    """Returns the time constant based on the sampling rate and timestr.

    timestr: 'us', 'ms', 's', 'm', 'h', 'd'
    """

    if unit == 'us':
        time_constant = sampling_rate / 1e6
    elif unit == 'ms':
        time_constant = sampling_rate / 1e3
    elif unit == 's':
        time_constant = sampling_rate
    elif unit == 'm':
        time_constant = sampling_rate * 60
    elif unit == 'h':
        time_constant = sampling_rate * 60 * 60
    elif unit == 'd':
        time_constant = sampling_rate * 60 * 60 * 24
    else:
        raise ValueError('timestr is wrong.')

    return float(time_constant)


def dtr(values, bits=16, min=-10, max=10):
    """Calculate real absolut values from dwords.

    """

    # Get the number of steps out of bits
    steps = 2 ** bits - 1

    # Process the data values and return numpy array
    values = [value * (max - min) / float((steps)) + min for value in values]
    return np.array(values)


def list_filenames(pathname):
    """List of sorted filenames in path, based on pattern.

    """
    files = glob.glob(pathname)
    return sorted(files)


def Hdf5File(*args, **kwargs):
    return h5py.File(*args, **kwargs)


class Hdf5Base(object):
    """Dynamic Hdf5 dataset class.

    """

    def __init__(self, dataset):
        self.__dict__['dataset'] = dataset
        self.__dict__['trim'] = True

    @classmethod
    def create(cls, hdf5_file, dataset_key, date=None, contact=None,
               comment=None, **dset_kwargs):
        """Create a new HDF5 dataset and initalize Hdf5Base.

        """

        if date is None:
            # Standart date format '2014/10/31 14:25:57'
            date = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        if comment is None:
            comment = ''
        if contact is None:
            contact = ''

        # Initalize Hdf5Base instance with new dataset
        hdf5base = cls(hdf5_file.create_dataset(dataset_key, **dset_kwargs))
        hdf5base.date = date
        hdf5base.comment = comment
        hdf5base.contact = contact

        # Return
        return hdf5base

    def __getitem__(self, key):

        # Handle floating point slice numbers
        if isinstance(key, slice):
            start = int(key.start) if key.start else None
            stop = int(key.stop) if key.stop else None
            step = int(key.step) if key.step else None

            # Pack new slice with integer values
            key = slice(start, stop, step)

        return self.dataset[key]

    def __dir__(self):
        return self.dataset.attrs.keys() + self.__dict__.keys()

    def __setitem__(self, key, value):
        self.dataset[key] = value

    def __getattr__(self, name):
        return self.dataset.attrs[name]

    def __setattr__(self, name, value):

        # First try to set class attribute otherwise set dataset attribute
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            if isinstance(value, str):
                # Trim lines
                if self.trim:
                    value = dedent(value)

            self.dataset.attrs[name] = value

    def __delattr__(self, name):
        del self.dataset.attrs[name]

    def __len__(self):
        """Number of levels.

        """
        return self.dataset.size

    @property
    def dtype(self):
        """Datatpye of the signal.

        """
        return self.dataset.dtype

    @property
    def shape(self):
        """Datatpye of the signal.

        """
        return self.dataset.shape

    def append(self, data):
        """Append new data at the end of signal.

        """

        data = np.array(data, dtype=self.dtype, copy=False)

        # Resize the dataset
        size0 = self.__len__()
        size1 = data.size + size0
        self.dataset.resize((size1,))

        # Insert new data
        self.dataset[size0:size1] = data

    def windows(self, length, start=0, stop=None, nr=None):
        """Iterator over windows of length.

        """

        # Set stop to the end of dataset
        if nr is not None:
            stop = int(start + nr * length)
        elif stop is None:
            stop = self.__len__()

        # Make everything integers of xrange and slice
        length = int(length)
        start = int(start)
        stop = int(stop)

        # Start iteration over data
        for position in xrange(start, stop, length):
            # Stop iteration if not enough datapoints available
            if stop < (position + length):
                return

            # Return current data window
            yield self.__getitem__(slice(position, position+length))


class Trace(Hdf5Base):

    @classmethod
    def create(cls, hdf_file, dataset_key, sampling_rate,
               dtype=np.dtype('float32'), shape=(0,), maxshape=(None,),
               chunks=(100000,)):

        trace = cls(Hdf5Base.create(hdf_file, dataset_key, dtype=dtype,
                    shape=shape, maxshape=maxshape, chunks=chunks).dataset)

        trace.sampling_rate = sampling_rate
        return trace

    def length(self, unit='m'):
        """Get the time length of the trace for unit.

        """
        if unit == 's':
            factor = 1
        elif unit == 'm':
            factor = 60
        elif unit == 'h':
            factor = 60 * 60

        return self.__len__() / float(self.sampling_rate * factor)

    def plot(self, start, stop, step=1, ax=None, **kwargs):

        if not ax:
            ax = plt.gca()

        x = np.arange(start, stop, step)
        y = self.__getitem__(slice(start, stop, step))

        return ax.plot(x, y, **kwargs)


class Detector(object):

    def __init__(self, average=1, nsigma=2, system=None, buffer=None):

        self._system = system

        if isinstance(average, int):
            self.average = average
        else:
            raise TypeError('average must be int')

        if isinstance(nsigma, (int, float)):
            self.nsigma = nsigma
        else:
            raise TypeError('nsigma must be int or foat')

        self._buffer = buffer

    @property
    def system(self):
        return self._system

    @system.setter
    def system(self, system):
        self._system = system

    @property
    def buffer(self):
        return self._buffer

    @buffer.setter
    def buffer(self, buffer):
        self._buffer = buffer

    def digitize(self, data, signal):
        """Digitize the the input data and store it in signal.

        """

        # Put buffer infront of input data
        try:
            data = np.concatenate([self._buffer, data])
        except ValueError:
            pass

        # Get the boundaries of the levels
        low0, high0, low1, high1 = self.abs

        # CYTHON: Digitize the data
        signal, self._buffer = _cycounting.digitize(data, signal,
                                                    int(self.average),
                                                    low0, high0, low1, high1)

        return signal

    @property
    def abs(self):
        """List of absolute boundarie values.

        The values are calculted from the level values with nsigma.
        """

        abs = []
        for level in self._system:
            low = level.center - level.sigma * self.nsigma
            high = level.center + level.sigma * self.nsigma
            abs += [low, high]
        return abs

    @property
    def rel(self):
        """List of absolute boundarie values.

        The values are calculted from the system levels with nsigma.

        """

        rel = []
        for level in self._system:
            rel += [level.center, level.sigma * self.nsigma]
        return rel

    def clear(self):
        """Clear the buffer.

        """
        self._buffer = None

    def plot(self, ax=None, **kwargs):

        if not ax:
            ax = plt.gca()

        lines = [plt.axhline(value, **kwargs)[0] for value in self.abs]
        return lines


class Level(object):

    def __init__(self, center, sigma):
        self.center = center
        self.sigma = abs(sigma)

    def __repr__(self):
        return 'Level(' + str(self.center) + ', ' + str(self.sigma) + ')'

    @property
    def low(self):
        return self.center - self.sigma

    @property
    def high(self):
        return self.center + self.sigma

    @property
    def rel(self):
        return (self.center, self.sigma)

    @property
    def abs(self):
        return (self.low, self.high)

    def plot(self, ax=None, inverted=False, **kwargs):

        if not ax:
            ax = plt.gca()
        else:
            plt.sca(ax)

        if inverted:
            lines = [plt.axhline(self.low, **kwargs),
                     plt.axhline(self.high, **kwargs)]
        else:
            lines = [plt.axvline(self.low, **kwargs),
                     plt.axvline(self.high, **kwargs)]

        return lines


class System(object):

    def __init__(self, *levels):
        self.levels = levels

    @classmethod
    def from_histogram(cls, histogram, start_parameters=None, levels=2, sigma=1):
        """Create System from Histogram.

        start_parameters: (a_0, mu_0 sigma_0, .... a_N, mu_N, sigma_N)
        """

        # Find start parameters from number of levels and noise width
        if start_parameters is None:
            start_parameters = list()

            # Get number of bins
            bins = histogram.bins
            freqs_n = histogram.freqs_n

            for peak in range(levels):
                # Get maximum and its position
                hight = np.max(freqs_n)
                center = np.mean(bins[freqs_n == hight])

                # Fit a normal distribution around the value
                fit = Fit(fnormal, bins, freqs_n, (hight, center, sigma))
                start_parameters.append(fit.parameters)

                center = fit.parameters[1]
                sigma = np.abs(fit.parameters[2])

                # Substrate fit from data
                freqs_n -= fit(bins)

                index = ((bins < (center - 2 * sigma)) | ((center + 2 * sigma) < bins))
                bins = bins[index]
                freqs_n = freqs_n[index]

        # Sort levels by position
        start_parameters = sorted(start_parameters, key=itemgetter(1))
        start_parameters = np.concatenate(start_parameters)
        #print start_parameters

        # Make a level fit
        fit = Fit(flevels, histogram.bins, histogram.freqs_n, start_parameters)

        # Filter levels=(mu_0, sigma_0, ..., mu_N, sigma_N)
        index = np.array([False, True, True] * (len(fit.parameters) / 3))
        levels = fit.parameters[index]
        system = cls(*[Level(levels[i], levels[i+1])
                       for i in xrange(0, len(levels), 2)])

        return system, fit

    def __getitem__(self, key):
        return self.levels[key]

    def __repr__(self):
        s = 'System:'
        for nr, level in enumerate(self.levels):
            s += '\n'
            s += str(nr) + ': ' + str(level)

        return s

    def __len__(self):
        return len(self.levels)

    @property
    def abs(self):
        values = []
        for level in self.levels:
            values += level.abs
        return values

    @property
    def rel(self):
        values = []
        for level in self.levels:
            values += level.rel
        return values

    @property
    def nr_of_levels(self):
        return self.__len__()

    def plot(self, ax=None, **kwargs):

        if not ax:
            ax = plt.gca()
        else:
            plt.sca(ax)

        lines = []
        for level in self.levels:
            lines += level.plot(ax, **kwargs)

        return lines


class LevelTrace(Hdf5Base):
    """Store and display fit parameters.

    Signal uses the h5py module to store the data in HDF5 format.

    """

    @classmethod
    def create(cls, hdf5_file, dataset_key, nr_of_levels, time_constant,
               date=None, contact=None, comment=None):

        # Define dtype
        type_list = list()
        for nr in range(nr_of_levels):
            type_list += [('hight' + str(nr), np.float32),
                          ('center' + str(nr), np.float32),
                          ('sigma' + str(nr), np.float32)]

        dtype = np.dtype(type_list)

        # Create dataset
        level_trace = cls(Hdf5Base.create(hdf5_file, dataset_key, date,
                                          contact, comment, shape=(0,),
                                          dtype=dtype,
                                          maxshape=(None,)).dataset)

        # Create signal instance and append undifined level
        level_trace.time_constant = time_constant
        level_trace.nr_of_levels = nr_of_levels
        return level_trace

    @property
    def keys(self):
        return self.dtype.names

    def fit(self, histogram, *system_arg, **system_kwargs):
        """Fit levels

        """
        # Fit with last parameters
        system, fit = System.from_histogram(histogram,
                                            *system_arg, **system_kwargs)

        # Create only positive numbers
        fit.parameters[2::3] = np.abs(fit.parameters[2::3])

        # Store parameters
        self.append(tuple(fit.parameters))

        # Return current system and fit
        return system, fit

    def plot(self, show, ax=None, **kwargs):

        # Put show strings into list
        show_list = list()
        if isinstance(show, str):
            show_list.append(show)
        else:
            show_list = show

        # Get current axes if not provided
        if not ax:
            ax = plt.gca()

        # Get y values
        ys = [self.__getitem__(key) for key in show_list]

        # Plot everything
        lines = [ax.plot(y, **kwargs)[0] for y in ys]

        return lines


class FFT(object):
    """Fast Fourier Transformation object.

    """

    def __init__(self, values=None, freqs=None, samples=0, sample_rate=1,
                 average=True):

        self.values = values
        self.freqs = freqs
        self.samples = samples
        self.sample_rate = float(sample_rate)
        self.average = average

    @classmethod
    def from_data(cls, data, sample_rate=1, average=True):
        """Create new FFT instance and transform data.

        """
        data = np.array(data, copy=False)
        values = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(data.size, d=1 / float(sample_rate))
        samples = data.size

        return cls(values, freqs, samples, sample_rate, average)

    @classmethod
    def from_hdf(cls, dataset):
        """Create FFT instance from dataset.

        """
        freqs = dataset['freqs']
        values = dataset['values']
        samples = dataset.attrs['samples']
        sample_rate = dataset.attrs['sample_rate']
        average = dataset.attrs['average']

        return cls(values, freqs, samples, sample_rate, average)

    def to_hdf(self, hdf5_file, dataset_key):
        # Create compound datatype
        fft_dtype = np.dtype([('freqs', self.freqs.dtype),
                              ('values', self.values.dtype)])

        # Fill array values with data
        ary = np.empty(shape=self.__len__(), dtype=fft_dtype)
        ary['freqs'] = self.freqs
        ary['values'] = self.values

        # Data to hdf5
        dset = hdf5_file.create_dataset(dataset_key, data=ary)
        dset.attrs['samples'] = self.samples
        dset.attrs['sample_rate'] = self.sample_rate
        dset.attrs['average'] = self.average

    def transform(self, data):
        data = np.array(data, copy=False)
        values = np.fft.rfft(data)

        try:
            self.values += values
            self.samples += data.size
        except TypeError:
            self.values = values
            self.freqs = np.fft.rfftfreq(data.size,
                                         d=1 / float(self.sample_rate))
            self.samples = data.size

    def __len__(self):
        return self.freqs.size

    @property
    def abs(self):
        return np.absolute(self.values)

    @property
    def abs_n(self):
        return self.abs / np.sqrt(self.samples)

    @property
    def real(self):
        return np.real(self.values)

    @property
    def real_n(self):
        return self.real / float(self.samples)

    @property
    def imag(self):
        return np.imag(self.values)

    @property
    def imag_n(self):
        return self.imag / float(self.samples)

    @property
    def angle(self):
        return np.angle(self.values)

    @property
    def power(self):
        return self.abs**2

    @property
    def power_n(self):
        return self.abs_n**2

    def plot(self, ax=None, show='abs_n', range=(5e3, np.inf),
             order=1e3, log=False, **kwargs):
        """Plot the fft spectrum in range.

        """
        if not ax:
            ax = plt.gca()

        # Get data
        values = self.__getattribute__(show)
        freqs = self.freqs / float(order)

        # Plot data range
        index = (self.freqs > range[0]) & (self.freqs < range[1])
        line = ax.plot(freqs[index], values[index], **kwargs)

        # Log everything
        if log:
            ax.set_yscale('log')

        # Set axis label
        ax.set_ylabel(show)
        ax.set_xlabel('frequency / ' + str(order))

        return line


class Signal(Hdf5Base):
    """Counting Signal class.

    Signal uses the h5py module to store the data in HDF5 format.

    """

    @classmethod
    def create(cls, hdf5_file, dataset_key,
               nr_of_levels, nsigma, average, date=None, contact=None,
               comment=None,
               state_type=np.int8,
               length_type=np.uint32,
               value_type=np.float32):

        # Define dtype
        dtype = np.dtype([('state', state_type),
                          ('length', length_type),
                          ('value', value_type)])

        # Initialize signal
        signal = cls(Hdf5Base.create(hdf5_file, dataset_key,
                                     date, contact, comment, shape=(0,),
                                     dtype=dtype, maxshape=(None,)).dataset)

        # Create signal instance and append undifined level
        signal.append((-1, 0, 0))
        signal.nr_of_levels = nr_of_levels
        signal.nsigma = nsigma
        signal.average = average
        return signal

    @property
    def keys(self):
        return self.dtype.names

    def plot(self):
        pass


class HistogramBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def freqs(self):
        pass

    @abc.abstractproperty
    def bins(self):
        pass

    @property
    def elements(self):
        """Return number of elements in histogram.

        """
        return self.freqs.sum()

    @property
    def freqs_n(self):
        """Return normed frequencies.

        """
        return self.freqs / float(self.elements)

    @property
    def values(self):
        """Return pair of bins and frequencies.

        """
        return self.bins, self.freqs

    @property
    def values_n(self):
        """Return pair of bins and normed frequencies.

        """
        return self.bins, self.freqs_n

    @property
    def mean(self):
        """Calculate mean value of histogram.

        """
        return np.sum(self.freqs * self.bins) / float(self.elements)

    @property
    def max_freq(self):
        """Return maximum of histogram.

        """
        return self.freqs.max()

    @property
    def max_freq_n(self):
        return self.freqs_n.max()

    def plot(self, ax=None, normed=True, inverted=False, **kwargs):

        if not ax:
            ax = plt.gca()

        y = self.freqs if not normed else self.freqs_n

        if not inverted:
            line = ax.plot(self.bins, y, **kwargs)
        else:
            line = ax.plot(y, self.bins, **kwargs)

        return line


class Histogram(HistogramBase):
    """Histogram class.

    """

    def __init__(self, bins=100, width=None, data=None):
        HistogramBase.__init__(self)

        if data is None:
            self._freqs = None
            self._bins = bins
            self._width = width
        else:
            self._freqs, self._bins = np.histogram(data, bins, width)

    def add(self, data):
        """Add data to the histogram.

        """
        try:
            self._freqs += np.histogram(data, self._bins)[0]
        except TypeError:
            self._freqs, self._bins = np.histogram(data, self._bins,
                                                   self._width)

    @property
    def bins(self):
        """Return bin values.

        """
        # Filter off empty frequency bins (necassary for fitting and plotting)
        return self._bins[:-1][self._freqs > 0]

    @property
    def freqs(self):
        """Return frequencies.

        """
        # Filter off empty frequency bins (necassary for fitting and plotting)
        return self._freqs[self._freqs > 0]


class Time(Histogram):

    def __init__(self, state, bins=1000, width=None, signal=None):

        self.state = state

        if not width:
            width = (0, bins)

        # Get all times of state from signal
        if signal is not None:
            times = signal['length'][signal['state'] == self._state]
            Histogram.__init__(self, bins, width, times)
        else:
            Histogram.__init__(self, bins=bins, width=width)

    def add(self, times):
        """Add time to signal.

        """

        if isinstance(times, Signal):
            times = times['length'][times['state'] == self.state]
        Histogram.add(self, times)

    def fit_exp(self, a=None, rate=None, range=None, normed=False):
        """Fit the time Histogram with an exponential function.

        """

        if rate is None:
            rate = -1 * self.mean

        if normed:
            if a is None:
                a = self.max_freq_n
            freqs = self.freqs_n
        else:
            if a is None:
                a = self.max_freq
            freqs = self.freqs

        bins = self.bins

        if range:
            index = (range[0] <= bins) & (bins <= range[1])
            bins = bins[index]
            freqs = freqs[index]

        fit = Fit.exp(bins, freqs, a, rate)
        fit.rate = np.abs(fit.parameters[-1])
        return fit

    def rate(self, sample_rate=500e3, range=None):
        """Rate extracted by the fit_exp method.

        """

        return sample_rate / np.abs(self.fit_exp(range=range).parameters[-1])

    def fft(self, samplerate=1):
        """Create FFT from frequencies.

        """
        return FFT.from_data(self.freqs, samplerate)

    def plot(self, ax=None, normed=False, log=True, **kwargs):
        """Plot time distribution.

        """
        line = Histogram.plot(self, ax, normed, False, **kwargs)

        if log:
            plt.yscale('log')

        return line


class Counter(HistogramBase):

    def __init__(self, state, delta, position=0, offset=0):
        HistogramBase.__init__(self)
        self._state = state
        self._histogram_dict = defaultdict(int)
        self._position = position
        self._delta = delta
        self._offset = offset
        self._counts = 0

    @property
    def offset(self):
        return self._offset

    @property
    def delta(self):
        return self._delta

    @property
    def bins(self):
        return np.array(self._histogram_dict.keys())

    @property
    def freqs(self):
        return np.array(self._histogram_dict.values())

    def count(self, signal):
        """Count the states for signal.

        """
        if isinstance(signal, Signal):
            positions = self._position + np.cumsum(signal['length'])
            signal = positions[signal['state'] == self._state]
        else:
            signal = self._position + signal

        self._position = signal[-1]

        # Count
        self._offset, self._counts = _cycounting.count(signal, self._delta,
                                                       self._offset,
                                                       self._counts,
                                                       self._histogram_dict)


class CallableList(list):

    def __init__(self, iterable):
        list.__init__(self, iterable)

    def __call__(self, *args, **kwargs):
        return [item(*args, **kwargs) for item in self.__iter__()]


class MultiBase(object):

    def __init__(self, instances, cls):
        # Avoid __setattr__ call
        self.__dict__['_instances'] = instances
        self.__dict__['_methods'] = dir(cls)

    def __dir__(self):
        return self._methods

    def __iter__(self):
        return iter(self._instances)

    def __getattr__(self, name):
        return CallableList([getattr(instance, name)
                             for instance in self._instances])

    def __setattr__(self, name, value):
        for instance in self._instances:
            setattr(instance, name, value)

    def __len__(self):
        return len(self._instances)

    def __getitem__(self, key):
        return self._instances[key]


class MultiDetector(MultiBase):

    def __init__(self, detectors):
        MultiBase.__init__(self, instances=detectors, cls=Detector)

    @classmethod
    def from_product(cls, average=[1], nsigma=[2], system=[None], factor=1):
        """Create MultiDetector from cartesian product of attributes.

        """
        return cls([Detector(*values)
                    for i in range(factor)
                    for values in product(average, nsigma, system)])

    def digitize(self, window, signals):
        if not len(signals) == self.__len__():
            raise ValueError('Signal len does not fit.')

        for detector, signal in zip(self.__iter__(), signals):
            detector.digitize(window, signal)


class MultiSignal(MultiBase):

    def __init__(self, signals):
        MultiBase.__init__(self, instances=signals, cls=Signal)

    @classmethod
    def from_product(cls, state, bin=[1000], width=[None], signal=[None],
                     nr=1):
        """Create MultiSignal from cartesian product of attributes.

        """
        return cls([Time(*values)
                    for i in range(nr)
                    for values in product(state, bin, width, signal)])


class MultiTime(MultiBase):

    def __init__(self, times):
        MultiBase.__init__(self, instances=times, cls=Time)

    @classmethod
    def from_product(cls, data=[None], start=[0], nr=1):
        """Create MultiTime from cartesian product of attributes.

        """
        return cls([Signal(*values)
                    for i in range(nr)
                    for values in product(data, start)])

    def add(self, signals):
        if not len(signals) == self.__len__():
            raise ValueError('Signal len does not fit.')

        for time, signal in zip(self.__iter__(), signals):
            time.add(signal)


def multi(detectors, nr_of_states=2):
    mdetector = MultiDetector(detectors)
    msignal = MultiSignal.from_product(nr=len(mdetector))
    return mdetector, msignal


def flinear(x, m=1, y0=0):
    """Linear function.

    """
    x = np.array(x, copy=False)
    return m * x + y0


def fexp(x, a=1, tau=-1):
    """Exponential function.
    """
    x = np.array(x, copy=False)
    return a * np.exp(x / float(tau))


def fnormal(x, a=1, mu=0, sigma=1):
    """Normal distribution.

    """
    x = np.array(x, copy=False)
    return a * np.exp(-(x - mu)**2 / (2. * sigma**2))


def flevels(x, *parameters):
    """Sum function of N differnt normal distributions.

    parameters: (a_0, mu_0 sigma_0, .... a_N, mu_N, sigma_N)
    """
    x = np.array(x, copy=False)

    # Create parameter triples (a_0, mu_0 sigma_0) ,... ,(a_N, mu_N, aigma_N)
    triples = (parameters[i:i+3] for i in range(0, len(parameters), 3))

    # (fnormal(x, a_0, mu_0 sigma_0) + ... + fnormal(x, a_0, mu_0 sigma_0)
    summands = (fnormal(x, *triple) for triple in triples)

    return np.sum(summands, 0)


class Fit(object):

    def __init__(self, function, xdata, ydata, start_parameters):
        self._function = function
        self._parameters, self._error = curve_fit(function, xdata, ydata,
                                                  start_parameters)

    @classmethod
    def linear(cls, xdata, ydata, m=1, y0=0):
        """Fit data with linear function.

        """
        return cls(flinear, xdata, ydata, (m, y0))

    @classmethod
    def exp(cls, xdata, ydata, a=1, tau=-1):
        """Fit data with exponential function.

        """
        return cls(fexp, xdata, ydata, (a, tau))

    @classmethod
    def normal(cls, xdata, ydata, a=1, mu=0, sigma=1):
        """Fit data with a normal distribution.

        """
        return cls(fnormal, xdata, ydata, (a, mu, sigma))

    def __call__(self, x):
        """Call fit function.

        """
        return self._function(x, *self._parameters)

    def values(self, x):
        """x and calculated y = Fit(x) values.

        """
        y = self._function(x, *self._parameters)
        return x, y

    @property
    def function(self):
        """Fit base function.

        """
        return self._function

    @property
    def parameters(self):
        """Fit parameters.

        """
        return self._parameters

    @property
    def error(self):
        """Fit error values.

        """
        return self._error

    def plot(self, x, ax=None, inverted=False, **kwargs):
        """Plot the fit function for x.

        """

        if not ax:
            ax = plt.gca()

        if not inverted:
            line = ax.plot(x, self.__call__(x), **kwargs)
        else:
            line = ax.plot(self.__call__(x), x, **kwargs)

        return line
