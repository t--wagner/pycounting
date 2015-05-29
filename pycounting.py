# -*- coding: utf-8 -*-

import abc
import os
import glob

from itertools import product
from collections import defaultdict, OrderedDict
from operator import itemgetter

import pickle as pickle
import datetime
from textwrap import dedent

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.optimize import curve_fit
from scipy.special import binom

import cycounting as _cycounting


def create_file(filename, override=False):
    """Create all directories and open new file.

    """

    # Create directory if it does not exit
    directory = os.path.dirname(filename)
    if directory:
        if not os.path.exists(directory):
            os.makedirs(directory)

    #  Check for existing file if overide is False
    if not override:
        if os.path.exists(filename):
            raise OSError('file exists.')

    # Return file object
    return open(filename, 'w')


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
    values = np.array(values, copy=False)

    # Get the number of steps out of bits
    steps = 2 ** bits - 1



    # Process the data values and return numpy array
    values = values * float(max - min) / float(steps) + min
    #values = [value * (max - min) / float((steps)) + min for value in values]
    return np.array(values)


def list_filenames(file_pattern):
    """List of sorted filenames in path, based on pattern.

    """
    files = glob.glob(file_pattern)
    return sorted(files)


def dict_filenames(file_pattern, index=0, seperator='_'):
    """Create ordered dictonary from filenames in based on pattern.

    """
    files = OrderedDict()
    for filename in list_filenames(file_pattern):
        basename = filename.split('/')[-1]
        key = basename.split(seperator)[index]
        files[key] = filename
    return files


def Hdf5File(*args, **kwargs):
    return h5py.File(*args, **kwargs)

def hdf_keys(filename):
    with h5py.File(filename,  mode='r') as hdf:
        return list(hdf.keys())

class CountingBase(object, metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __getitem__(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

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
        for position in range(start, stop, length):
            # Stop iteration if not enough datapoints available
            if stop < (position + length):
                return

            # Return current data window
            yield self.__getitem__(slice(position, position+length))

class Hdf5Base(CountingBase):
    """Dynamic Hdf5 dataset class.

    """

    def __init__(self, dataset, hdf_file=None, *file_args, **file_kwargs):

        if hdf_file:
            # Create or open hdf file
            if isinstance(hdf_file, str):
                hdf_file = Hdf5File(hdf_file, *file_args, **file_kwargs)

            # Get the dataset object for hdf file
            dataset = hdf_file[dataset]

        CountingBase.__init__(self)
        self.__dict__['dataset'] = dataset
        self.__dict__['trim'] = True

    # Context manager
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @classmethod
    def create(cls, dataset, hdf_file , date=None, contact=None,
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

        if isinstance(hdf_file, str):
            hdf_file = Hdf5File(hdf_file)

        # Initalize Hdf5Base instance with new dataset
        hdf5base = cls(hdf_file.create_dataset(dataset, **dset_kwargs))
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
        return list(self.dataset.attrs.keys()) + list(self.__dict__.keys())

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

    def close(self):
        """Close file instance in which the dataset resides.

        """
        self.dataset.file.close()

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

    def extend(self, data):
        """Append new data at the end of signal.

        """

        data = np.array(data, dtype=self.dtype, copy=False)

        # Resize the dataset
        size0 = self.__len__()
        size1 = data.size + size0
        self.dataset.resize((size1,))

        # Insert new data
        self.dataset[size0:size1] = data


class CallableList(list):

    def __init__(self, iterable):
        list.__init__(self, iterable)

    def __call__(self, *args, **kwargs):
        return [item(*args, **kwargs) for item in self.__iter__()]


class MultiBase(object):

    def __init__(self, cls, instances=[]):

        # Avoid __setattr__ call
        self.__dict__['_instances'] = list(instances)
        self.__dict__['_methods'] = dir(cls)

    def __dir__(self):
        return self._methods

    def __iter__(self):
        return iter(self.__dict__['_instances'])

    def __getattr__(self, name):
        """Return a callable list.

         First lookup in MultiClass and later in the instances.

        """
        return CallableList([getattr(instance, name)
                            for instance in self.__iter__()])

    def __setattr__(self, name, value):
        """Get attribute.

        First lookup in MultiClass and later in the instances.

        """
        for instance in self.__dict__['_instances']:
            setattr(instance, name, value)

    def __len__(self):
        """x.__len__(key) <==> len(x)

        """
        return len(self._instances)

    def __getitem__(self, key):
        """x.__getitem__(key) <==> x[key]

        """
        instances = self._instances[key]

        if isinstance(key, slice):
            instances = self.__class__(instances)

        return instances

    def append(self, object):
        self.__dict__['_instances'].append(object)

    def extend(self, iterable):
        for element in iterable:
            self.append(element)

    def to_pickle(self, file, override=False):
        """Pcikle instance to file.

        """

        if isinstance(file, str):
            with create_file(file, override) as fobj:
                pickle.dump(self, fobj)
        else:
            pickle.dump(self, file)

    @classmethod
    def from_pickle(cls, file):
        """Unpickle from file.

        """
        if isinstance(file, str):
            # Handle filenames
            with open(file, 'r') as fobj:
                multi = pickle.load(fobj)
        else:
            # Handle file objects
            multi = pickle.load(file)

        return multi

    def __setstate__(self, dict):
        """For correct unpickling.

        Otherwise pickle.load will cause an error because of __getattr__
        """
        self.__dict__.update(dict)   # update attributes

    def sort(self, mask):
        """Sort by delta

        """
        if isinstance(mask, str):
            mask = self.__getattr__(mask)

        self._instances.sort(key=dict(list(zip(self._instances, mask))).get)

    def get(self, value, attribute):
        matches = [instance for instance in self.__iter__()
                   if instance.__getattribute__(attribute) == value]

        if len(matches) == 0:
            return None
        elif len(matches) == 1:
            return matches[0]
        else:
            return self.__class__(matches)


class Trace(Hdf5Base):

    @classmethod
    def create(cls, dataset, hdf_file, sampling_rate,
               dtype=np.dtype('float32'), shape=(0,), maxshape=(None,),
               chunks=(100000,)):

        trace = cls(Hdf5Base.create(dataset, hdf_file, dtype=dtype,
                    shape=shape, maxshape=maxshape, chunks=chunks).dataset)

        trace.sampling_rate = sampling_rate
        return trace

    def duration(self, unit='m'):
        """Get the time length of the trace in unit.

        """
        if unit == 's':
            factor = 1
        elif unit == 'm':
            factor = 60
        elif unit == 'h':
            factor = 60 * 60

        return self.__len__() / float(self.sampling_rate * factor)

    def _time_to_index(self, start, stop, step=None):

        start *= self.sampling_rate
        stop  *= self.sampling_rate

        if step:
            step  *= self.sampling_rate
        else:
            step = 1

        return (int(start), int(stop), int(step))

    def xdata(self, start, stop, step=None, time=True):

        if time:
            start, stop, step = self._time_to_index(start, stop, step)
            x = np.arange(start, stop, step) / self.sampling_rate
        else:
            if not step:
                step = 1
            x = np.arange(int(start), int(stop), int(step))

        return x

    def ydata(self, start, stop, step=None, time=True):

        if time:
            start, stop, step = self._time_to_index(start, stop, step)

        return self.__getitem__(slice(start, stop, step))

    def data(self, start, stop, step=None, time=False):

        x = self.xdata(start, stop, step, time)
        y = self.ydata(start, stop, step, time)

        return x, y

    def plot(self, start, stop, step=None, ax=None, time=True, **plt_kwargs):

        # Get current axes
        if not ax:
            ax = plt.gca()

        # Get data
        x, y = self.data(start, stop, step, time)

        mpl_line2d, = ax.plot(x, y, **plt_kwargs)

        return mpl_line2d


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

    def plot(self, ax=None, inverted=False, **kwargs):

        if not ax:
            ax = plt.gca()

        if inverted:
            mpl_lines2d = [plt.axhline(value, **kwargs) for value in self.abs]
        else:
            mpl_lines2d = [plt.axvline(value, **kwargs) for value in self.abs]

        return mpl_lines2d


class MultiDetector(MultiBase):

    def __init__(self, detectors):
        MultiBase.__init__(self, cls=Detector, instances=detectors)

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


class Level(object):

    def __init__(self, center, sigma):
        self.center = center
        self.sigma = abs(sigma)

    @classmethod
    def from_abs(cls, low, high):
        sigma = float(high - low) / 2
        center = low + sigma
        return cls(center, sigma)

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
            mpl_lines2d = [plt.axhline(self.low, **kwargs),
                           plt.axhline(self.high, **kwargs)]
        else:
            mpl_lines2d = [plt.axvline(self.low, **kwargs),
                           plt.axvline(self.high, **kwargs)]

        return mpl_lines2d


class System(object):

    def __init__(self, *levels):
        self.levels = levels

    @classmethod
    def from_tuples(cls, *tuples):
        levels = [Level(center, sigma) for center, sigma in tuples]
        return cls(*levels)

    @classmethod
    def from_tuples_abs(cls, *tuples):
        levels = (Level.from_abs(high, low) for high, low in tuples)
        return cls(*levels)

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
                       for i in range(0, len(levels), 2)])

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

        mpl_lines2d = []
        for level in self.levels:
            mpl_lines2d += level.plot(ax, **kwargs)

        return mpl_lines2d


class LevelTrace(Hdf5Base):
    """Store and display fit parameters.

    Signal uses the h5py module to store the data in HDF5 format.

    """

    @classmethod
    def create(cls, dataset, hdf_file, nr_of_levels, time_constant,
               date=None, contact=None, comment=None):

        # Define dtype
        type_list = list()
        for nr in range(nr_of_levels):
            type_list += [('hight' + str(nr), np.float32),
                          ('center' + str(nr), np.float32),
                          ('sigma' + str(nr), np.float32)]

        dtype = np.dtype(type_list)

        # Create dataset
        level_trace = cls(Hdf5Base.create(dataset, hdf_file, date,
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
        self.extend(tuple(fit.parameters))

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
        x = self.len() * self.time_constant
        ys = [self.__getitem__(key) for key in show_list]

        # Plot everything
        mpl_lines2d = [ax.plot(x, y, **kwargs)[0] for y in ys]

        return mpl_lines2d


class FFT(object):
    """Fast Fourier Transformation object.

    """

    def __init__(self, values=None, freqs=None, samples=0, sampling_rate=1,
                 average=True):

        self.values = values
        self.freqs = freqs
        self.samples = samples
        self.sampling_rate = float(sampling_rate)
        self.average = average

    @classmethod
    def from_data(cls, data, sampling_rate=1, average=True):
        """Create new FFT instance and transform data.

        """
        data = np.array(data, copy=False)
        values = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(data.size, d=1 / float(sampling_rate))
        samples = data.size

        return cls(values, freqs, samples, sampling_rate, average)

    def transform(self, data):
        data = np.array(data, copy=False)
        values = np.fft.rfft(data)

        try:
            self.values += values
            self.samples += data.size
        except TypeError:
            self.values = values
            self.freqs = np.fft.rfftfreq(data.size,
                                         d=1 / float(self.sampling_rate))
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
        mpl_line2d, = ax.plot(freqs[index], values[index], **kwargs)

        # Log everything
        if log:
            ax.set_yscale('log')

        # Set axis label
        ax.set_ylabel(show)
        ax.set_xlabel('frequency / ' + str(order))

        return mpl_line2d


class Signal(CountingBase):

    _key_state = 'state'
    _key_length = 'length'
    _key_value = 'value'

    def __init__(self, data, start=0):
        CountingBase.__init__(self)

        self.data = data
        self.start = start

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Signal(self.data[key])
        else:
            return self.data[key]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)

    def __str__(self):
        return str(self.data)

    @property
    def state(self):
        return self.data[self._key_state]

    @property
    def length(self):
        return self.data[self._key_length]

    @property
    def value(self):
        return self.data[self._key_value]

    @property
    def position(self):
        return np.cumsum(self.start + self.length)

    def plot(self, state=False, ax=None, **kwargs):

        if not ax:
            ax = plt.gca()

        x = self.position

        if state:
            y = self.state
        else:
            y = self.value

        mpl_line2d, = ax.step(x, y, **kwargs)

        return mpl_line2d


class SignalFile(Hdf5Base):
    """Counting Signal class.

    Signal uses the h5py module to store the data in HDF5 format.

    """

    def __getitem__(self, key):

        signal = Hdf5Base.__getitem__(self, key)

        if isinstance(key, slice):
            signal = Signal(signal)

        return signal

    @classmethod
    def create(cls, dataset, hdf5_file,
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
        signal = cls(Hdf5Base.create(dataset, hdf5_file,
                                     date, contact, comment, shape=(0,),
                                     dtype=dtype, maxshape=(None,)).dataset)

        # Create signal instance and append undifined level
        signal.extend((-1, 0, 0))
        signal.nr_of_levels = nr_of_levels
        signal.nsigma = nsigma
        signal.average = average
        return signal

    @property
    def keys(self):
        return self.dtype.names

    def plot(self):
        pass


class MultiSignal(MultiBase):

    def __init__(self, signals):
        MultiBase.__init__(self, cls=Signal, instances=signals)

    @classmethod
    def from_product(cls, state, bin=[1000], width=[None], signal=[None],
                     nr=1):
        """Create MultiSignal from cartesian product of attributes.

        """
        return cls([Signal(*values)
                    for i in range(nr)
                    for values in product(state, bin, width, signal)])


class HistogramBase(object, metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def histogram(self):
        pass

    @property
    def bins(self):
        return self.histogram[0]

    @property
    def freqs(self):
        return self.histogram[1]

    @property
    def items(self):
        return list(zip(self.bins, self.freqs))

    def __iter__(self):
        return zip(self.bins, self.freqs)

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
    def mean(self):
        """Calculate mean value of histogram.

        """
        #return self.moment(1)
        return np.sum(self.freqs * self.bins) / float(self.elements)

    @property
    def variance(self):
        # The second central moment is the variance
        return self.moment_central(2)

    @property
    def standard_deviation(self):
        # The square root of the variance
        return np.sqrt(self.variance)

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
            line, = ax.plot(self.bins, y, **kwargs)
        else:
            line, = ax.plot(y, self.bins, **kwargs)

        return line


    def moment(self, n, c=0):
        """Calculate the n-th moment of histogram about the value c.

        """

        # Make sure teh bins are float type
        bins = np.array(self.bins, dtype=np.float, copy=False)
        moment = np.sum(self.freqs * ((bins - c) ** n)) / self.elements
        return moment

    def moment_central(self, n):
        """Calculate the n-th central moment of histogram.

        """
        return self.moment(n, self.mean)

    def cumulants(self, n, return_moments=False):

        moments = [self.moment(i) for i in range(n + 1)]
        if return_moments:
            return fcumulants(moments), moments
        else:
            return fcumulants(moments)


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

    def fill(self, iteratable):
        """Add data from iteratable to the histogram.

        """

        for data in iteratable:
            self.add(data)

    @property
    def histogram(self):
        index = self._freqs > 0
        bins = self._bins[:-1][index]
        freqs = self._freqs[index]

        return bins, freqs


class Time(Histogram):

    def __init__(self, state, bins=1000, width=None, signal=None):

        self._state = state

        if not width:
            width = (0, bins)

        # Get all times of state from signal
        if signal is not None:
            times = signal.length[signal.state == self._state]
            Histogram.__init__(self, bins, width, times)
        else:
            Histogram.__init__(self, bins=bins, width=width)

    @property
    def state(self):
        return self._state

    def add(self, times):
        """Add times.

        """

        if isinstance(times, (Signal, SignalFile)):
            times = times.length[times.state == self.state]
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

    def fft(self, sampling_rate=1):
        """Create FFT from frequencies.

        """
        return FFT.from_data(self.freqs, sampling_rate)

    def plot(self, ax=None, normed=False, log=True, **kwargs):
        """Plot time distribution.

        """
        if not ax:
            ax = plt.gca()

        line = Histogram.plot(self, ax, normed, **kwargs)

        if log:
            ax.set_yscale('log')

        return line


class MultiTime(MultiBase):

    def __init__(self, times):
        MultiBase.__init__(self, cls=Time, instances=times)

    @classmethod
    def from_states(cls, states, *time_args, **time_kwargs):
        """Create MultiTime from range of states.

        """
        return cls([Time(state=state, *time_args, **time_kwargs) for state in states])

    def fill(self, iteratable):
        for signals in iteratable:
            self.add(signals)

    def get(self, value, attribute='state'):
        return MultiBase.get(self, value, attribute)


class CounterTraceFile(object):
    pass


class CounterTrace(Hdf5Base):
    """Counter Trace.

    """

    @classmethod
    def create(cls, dataset, hdf_file, state, delta, date=None, contact=None,
               comment=None, **dset_kwargs):

        # Define Datatype
        dtype = np.dtype(np.int16)

        # Create HdfFile
        counter_trace = cls(Hdf5Base.create(dataset, hdf_file, date,
                                            contact, comment, shape=(0,),
                                            dtype=dtype,
                                            maxshape=(None,)).dataset)

        # Set attributes
        counter_trace.state = state
        counter_trace.delta = delta

        # Return initalized instance
        return counter_trace

    def count(self, signal):
        """Count the states for signal.

        """
        if isinstance(signal, (Signal, SignalFile)):
            positions = self._position + np.cumsum(signal['length'])
            signal = positions[signal['state'] == self._state]
        else:
            signal = self._position + signal

        self._position = signal[-1]

        # Count
        self._offset, self._counts, trace = _cycounting.count2(signal,
                                                        self._delta,
                                                        self._offset,
                                                        self._counts)

        return trace


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
    def histogram(self):

        histogram = list(zip(*sorted(self._histogram_dict.items())))
        return np.array(histogram[0]), np.array(histogram[1])

    def count(self, signal):
        """Count the states for signal.

        """
        if isinstance(signal, (Signal, SignalFile)):
            positions = self._position + np.cumsum(signal['length'])
            signal = positions[signal['state'] == self._state]
        else:
            signal = self._position + signal

        self._position = signal[-1]

        # Count
        self._offset, self._counts = _cycounting.count(signal,
                                                       self._delta,
                                                       self._offset,
                                                       self._counts,
                                                       self._histogram_dict)


class MultiCounter(MultiBase):

    def __init__(self, counters):
        MultiBase.__init__(self, cls=Counter, instances=counters)

    @classmethod
    def from_deltas(cls, state, deltas):
        return cls([Counter(state, delta) for delta in deltas])

    @classmethod
    def from_range(cls, state, start, stop, step):
        deltas = np.arange(start, stop, step)
        return cls.from_deltas(state, deltas)

    @classmethod
    def from_linspace(cls, state, start, stop, points):
        deltas = np.linspace(start, stop, points)
        return cls.from_deltas(state, deltas)

    def sort(self, mask='delta'):
        MultiBase.sort(self, mask)

    def get(self, value, attribute='delta'):
        return MultiBase.get(self, value, attribute)


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


def fcumulants(moments, n=None):
    """Calculate the corresponding moments from cumulants.

    """

    cumulants = []

    if n is None:
        n = len(moments)
    else:
        n = int(n) + 1

    for m in range(n):
        cumulants.append(moments[m])
        for k in range(m):
            cumulants[m] -= binom(m - 1, k - 1) * cumulants[k] * moments[m - k]

    return cumulants


# Long time calculations
def a(tau_in, tau_out):
    return (tau_in - tau_out) / float(tau_in + tau_out)


def c1(t, tau_in, tau_out):
    return tau_in * tau_out / float(tau_in + tau_out) * t


def c2(t, tau_in, tau_out):
    return 1/2. * (1 + a(tau_in, tau_out)**2) * c1(t, tau_in, tau_out)


def c2_n(t, tau_in, tau_out):
    return 1/2. * (1 + a(tau_in, tau_out)**2)
