# -*- coding: utf-8 -*-

from os.path import getsize
import numpy as np
from scipy.optimize import curve_fit
import glob
import matplotlib.pyplot as plt
from cdetector import digitize as _digitize
from itertools import product
#import h5py


def tc(sampling_rate, timestr='s'):
    """Returns the time constant based on the sampling rate and timestr.

    timestr: 'us', 'ms', 's', 'm', 'h', 'd'
    """

    if timestr == 'us':
        time_constant = sampling_rate / 1e6
    elif timestr == 'ms':
        time_constant = sampling_rate / 1e3
    elif timestr == 's':
        time_constant = sampling_rate
    elif timestr == 'm':
        time_constant = sampling_rate * 60
    elif timestr == 'h':
        time_constant = sampling_rate * 60 * 60
    elif timestr == 'd':
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


class Hdf5Base(object):

    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, key):
        return self._dataset[key]

    def __setitem__(self, key, value):
        self._dataset[key] = value

    def __len__(self):
        """Number of levels.

        """
        return self._dataset.size

    def windows(self, length, start=0, stop=None):
        """Iterator over windows of length.

        """

        # set stop to the end of dataset
        if stop is None:
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

    @property
    def chunks(self):
        return self._dataset.chunks

    @property
    def dtype(self):
        """Datatpye of the signal.

        """
        return self._dataset.dtype

    @property
    def keys(self):
        return self.dtype.names

    @property
    def attrs(self):
        """Access the attributes.

        """
        return self._dataset.attrs

    def append(self, data):
        """Append new data at the end of signal.

        """

        data = np.array(data, dtype=self.dtype, copy=False)

        # Resize the dataset
        size0 = self.__len__()
        size1 = data.size + size0
        self._dataset.resize((size1,))

        # Insert new data
        self._dataset[size0:size1] = data


def trace_to_hdf(trace, hdf_file, dataset_string, length=100e3):

    length = int(length)
    dset = hdf_file.create_dataset(dataset_string, dtype=trace._dtype,
                                   shape=(len(trace),))

    position = 0
    for window in trace.windows(length, start=0, stop=len(trace)):
        dset[position:position+length] = window
        position += length


class Trace(object):
    """Handle binary counting data files very flexible.

    """

    def __init__(self, files, datatype, samplerate):

        self._samplerate = samplerate

        if isinstance(files, (list, tuple)):
            self._files = list(files)
        else:
            self._files = list_filenames(files)

        # Open all datafiles in readable binary mode
        self._fileobjs = []
        for filename in self._files:
            self._fileobjs.append(open(filename, 'rb'))

        # Set position to first file
        self._fileobj = self._fileobjs[0]

        # Store the datatype information
        self._datatype = datatype

        if self._datatype in ['int', 'int32']:
            self._dtype = np.dtype(np.int32)
        elif self._datatype == 'ushort':
            self._dtype = np.dtype(np.ushort)
        else:
            self._dtype = self._datatype
            # raise TypeError('Unsupported datatype')

        self._datapointsize = self._dtype.itemsize

        # Calculate the size in bytes of all files
        self._datasize = sum(getsize(fname) for fname in self._files)

        # Calculate the start and stop positions the files
        self._file_start_positions = []
        self._file_stop_positions = []
        position = 0
        for filename in self._files:
            self._file_start_positions.append(position)
            position += getsize(filename) / self._datapointsize
            self._file_stop_positions.append(position)

    def __del__(self):
        """Close all datafiles.

        """
        for fileobj in self._fileobjs:
            fileobj.close()

    def __len__(self):
        """The number of datapoints.

        """
        return self._file_stop_positions[-1]

    def __getitem__(self, key):
        """Return numpy array between start and stop.

        """
        if isinstance(key, slice):
            # Unpack slice and handle values in slice
            start = int(key.start) if key.start else 0
            stop = int(key.stop) if key.stop else self.__len__()
            step = int(key.step) if key.step else 1

            # Handle negative positions
            start = self._tpos(start)
            stop = self._tpos(stop)

            # Set position
            self.position = int(start)

            # Return window
            return self.next(stop - start)[::step]
        elif isinstance(key, int):
            self.position = key
            return self.next()
        else:
            raise TypeError('Trace indices must be integers, not ' +
                            type(key).__name__)

    def __iter__(self):
        """Iterator over all trace.

        """
        self.position = 0
        return self

    @property
    def files(self):
        """List of filenames.

        """
        return self._files

    def datasize(self, unit='gb'):
        """The size of all files in MBytes.

        """
        if unit == 'gb':
            divider = 1024 * 1024 * 1024
        elif unit == 'mb':
            divider = 1024 * 1024
        elif unit == 'kb':
            divider = 1024
        elif unit == 'b':
            divider = 1

        return self._datasize / float(divider)

    @property
    def datatype(self):
        """The datatype.

        """
        return self._datatype

    def _tpos(self, position):
        """Handle negative positions and check range.

        Use this internal to handle input.
        """

        # Tranform negative position value
        if position < 0:
            position += self._file_stop_positions[-1]

        # Raise IndexError if out of range
        if ((position < 0) or (position > self._file_stop_positions[-1])):
            raise IndexError('position out of range')

        return position

    @property
    def position(self):
        """Return the current datapoint position.

        """

        # Get the file index
        file_index = self._fileobjs.index(self._fileobj)

        # Get the start position of the file and current position
        file_start_position = self._file_start_positions[file_index]
        file_current_position = self._fileobj.tell() / self._datapointsize

        return file_start_position + file_current_position

    @position.setter
    def position(self, position):
        """Set the current datapoint position.

        """

        # Make sure the position is valid positive number
        position = self._tpos(position)

        # Check if position is valid otherwise raise IndexError
        if ((position < 0) or (position > self._file_stop_positions[-1])):
            raise IndexError('position out of range')

        # Find the file wich belongs to the position
        for index, stopposition in enumerate(self._file_stop_positions):
            if position < stopposition:
                # Set the corresponding fileobj
                self._fileobj = self._fileobjs[index]
                break

        # Set the new positon
        datapoint_nr = position - self._file_start_positions[index]
        file_position = datapoint_nr * self._datapointsize
        self._fileobj.seek(file_position, 0)

    def next(self, length=1):
        """Return np.array with next datapoints of trace.

        """

        # Every other data method uses this next_window method

        # Get the array of length
        length = int(length)
        data = np.fromfile(self._fileobj, self._dtype, length)

        # Check array size to find end of file
        rest = length - data.size
        if rest:
            # Update position for next file
            self.position = self.position
            data = np.append(data,
                             np.fromfile(self._fileobj, self._dtype, rest))

        return data

    def windows(self, length, start=0, stop=None, nr=None):
        """Iterator over windows of length.

        """

        if start is not None:
            start = self._tpos(start)
            self.position = int(start)

        if stop is None:
            stop = self._file_stop_positions[-1]
        stop = self._tpos(stop)

        if nr is not None:
            # Window number defined
            for nr in xrange(nr):
                yield self.next(length)
        else:
            # Stop position defined
            while self.position + length < stop:
                yield self.next(length)

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
        self._buffer = _digitize(data, signal, int(self.average),
                                 low0, high0, low1, high1)

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
    def from_histogram(cls, histogram, start_parameters):
        """Create System from Histogram.

        start_parameters: (a_0, mu_0 sigma_0, .... a_N, mu_N, sigma_N)
        """

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
    def create(cls, hdf5_file, key, nr_of_levels, time_constant):

        # Define dtype
        type_list = list()
        for nr in range(nr_of_levels):
            type_list += [('hight' + str(nr), np.float32),
                          ('center' + str(nr), np.float32),
                          ('sigma' + str(nr), np.float32)]
        dtype = np.dtype(type_list)

        # Create dataset
        dset = hdf5_file.create_dataset(key, shape=(0,),
                                        dtype=dtype, maxshape=(None,))

        # Create signal instance and append undifined level
        level_trace = cls(dset)
        level_trace.attrs['time_constant'] = time_constant
        return level_trace

    @property
    def time_constant(self):
        return self.attrs['time_constant']

    @time_constant.setter
    def time_constant(self, time_constant):
        self.attrs['time_constant'] = time_constant

    def fit(self, histogram, start_parameters):
        """Fit levels

        """
        # Fit with last parameters
        system, fit = System.from_histogram(histogram, start_parameters)

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


class Signal(Hdf5Base):
    """Counting Signal class.

    Signal uses the h5py module to store the data in HDF5 format.

    """

    @classmethod
    def create(cls, hdf5_file, dataset_string, state_type=np.int8,
               length_type=np.uint32, value_type=np.float32, **kwargs):

        # Define dtype
        dtype = np.dtype([('state', state_type),
                          ('length', length_type),
                          ('value', value_type)])

        # Create dataset
        dset = hdf5_file.create_dataset(dataset_string, shape=(0,),
                                        dtype=dtype, maxshape=(None,),
                                        **kwargs)

        # Create signal instance and append undifined level
        signal = cls(dset)
        signal.append((-1, 0, 0))
        return signal


class Histogram(object):
    """Histogram class.

    """

    def __init__(self, bins=1000, width=None, data=None):
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

    @property
    def elements(self):
        """Return number of elements in histogram.

        """
        return self._freqs.sum()

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

    @property
    def freqs_n(self):
        """Return normed frequencies.

        """
        return self.freqs / float(self.elements)

    @property
    def values(self):
        """Return pair of bins and frequencies."""
        return self.bins, self.freqs

    @property
    def values_n(self):
        """Return pair of bins and normed frequencies."""
        return self.bins, self.freqs_n

    def plot(self, ax=None, normed=True, inverted=False, **kwargs):

        if not ax:
            ax = plt.gca()

        y = self.freqs if not normed else self.freqs_n

        if not inverted:
            line = ax.plot(self.bins, y, **kwargs)
            # line = ax.step(self.bins, y, where='mid', **kwargs)
        else:
            line = ax.plot(y, self.bins, **kwargs)
            # line = ax.step(y, self.bins, where='mid', **kwargs)

        return line


class Time(Histogram):

    def __init__(self, state, bins=1000, width=None, signal=None):

        self.state = state

        if not width:
            width = (0, bins)

        # Get all times of state from signal
        if signal is not None:
            times = signal['length'][signal.state == self._state]
            Histogram.__init__(self, bins, width, times)
        else:
            Histogram.__init__(self, bins=bins, width=width)

    def add(self, signal):
        """Add time to signal.

        """
        times = signal['length'][signal.state == self.state]
        Histogram.add(self, times)

    def fit_exp(self, a=None, rate=None, normed=True):
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

        fit = Fit.exp(self.bins, freqs, a, rate)
        fit.rate = np.abs(fit.parameters[-1])
        return fit

    @property
    def rate(self):
        """Rate extracted by the fit_exp method.

        """
        return np.abs(self.fit_exp().parameters[-1])

    def fft(self, samplerate=1):
        """Create FFT from frequencies.

        """
        return FFT(self.freqs, samplerate)

    def plot(self, ax=None, normed=True, log=True, **kwargs):
        """Plot time distribution.

        """
        line = Histogram.plot(self, ax, normed, False, **kwargs)

        if log:
            plt.yscale('log')

        return line


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


class FFT(object):

    def __init__(self, data=None, samplerate=1):

        self._samplerate = samplerate

        if data is not None:
            self.transform(data)

    def transform(self, data):
        data = np.array(data, copy=False)
        fft = np.fft.rfft(data)

        try:
            self._fft += fft
            self._samples += data.size
        except AttributeError:
            self._fft = fft
            self._freq = np.fft.rfftfreq(data.size,
                                         d=1/float(self._samplerate))
            self._samples = data.size

    @property
    def samplerate(self):
        return self._samplerate

    @property
    def samples(self):
        return self._samples

    @property
    def freq(self):
        return self._freq

    @property
    def abs(self):
        return np.absolute(self._fft)

    @property
    def abs_n(self):
        return self.abs / np.sqrt(self._samples)

    @property
    def real(self):
        return np.real(self._fft)

    @property
    def real_n(self):
        return self.real / float(self._samples)

    @property
    def imag(self):
        return np.imag(self._fft)

    @property
    def imag_n(self):
        return self.imag / float(self._samples)

    @property
    def angle(self):
        return np.angle(self._fft)

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
        fft = self.__getattribute__(show)
        freq = self._freq / float(order)

        # Plot data range
        index = (self.freq > range[0]) & (self.freq < range[1])
        line = ax.plot(freq[index], fft[index], **kwargs)

        # Log everything
        if log:
            ax.set_yscale('log')

        # Set axis label
        ax.set_ylabel(show)
        ax.set_xlabel('frequency / ' + str(order))

        return line


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
