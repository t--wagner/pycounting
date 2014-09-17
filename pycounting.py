# -*- coding: utf-8 -*-

import os
import numpy as np
from collections import defaultdict
from scipy.optimize import curve_fit


class Trace(object):
    """Handle binary counting data files very flexible.

    """

    def __init__(self, filelist, datatype):

        # Store the filelist
        self._filelist = filelist

        # Open all datafiles in readable binary mode
        self._fileobjs = []
        for filename in self._filelist:
            self._fileobjs.append(open(filename, 'rb'))

        # Set position to first file
        self._fileobj = self._fileobjs[0]

        # Store the datatype information
        self._datatype = datatype

        if self._datatype in ['int', 'int32']:
            self._dtype = np.dtype(np.int32)
        elif self._datatype == 'ushort':
            self._dtype = ''
        else:
            raise TypeError('Unsupported datatype')

        self._datapointsize = self._dtype.itemsize

        # Calculate the size in bytes of all files
        self._datasize = 0
        for filename in self._filelist:
            self._datasize += os.path.getsize(filename)

        # Calculate the start and stop positions the files
        self._file_start_positions = []
        self._file_stop_positions = []
        position = 0
        for filename in self._filelist:
            self._file_start_positions.append(position)
            position += os.path.getsize(filename) / self._datapointsize
            self._file_stop_positions.append(position)

        self._datapoint = 0

    def __del__(self):
        """Close all datafiles.

        """
        for fileobj in self._fileobjs:
            fileobj.close()

    def __len__(self):
        return self.points

    def __getitem__(self, key):
        self.position = key
        return self.next()

    def __iter__(self):
        self.position = 0
        return self

    @property
    def filelist(self):
        """Return list of filenames.

        """
        return self._filelist

    @property
    def datasize(self):
        """Return the size of all files in kBytes.

        """
        return self._datasize

    @property
    def datatype(self):
        """Return the datatype.

        """
        return self._datatype

    @property
    def points(self):
        """Return the number of datapoints in all files.

        """
        return self._file_stop_positions[-1]

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

    def _get_position(self, position):

        if position < 0:
            position += self._file_stop_positions[-1]

        # Raise IndexError if out of range
        if ((position < 0) or (position > self._file_stop_positions[-1])):
            raise IndexError('position out of range')

        return position

    @position.setter
    def position(self, position):
        """Set the current datapoint position.

        """

        # Make sure the position is valid positive number
        position = self._get_position(position)

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

    def next(self):
        """Return next datapoint of trace.

        """

        datapoint, = self.next_window(1)
        return datapoint

    def next_window(self, length):
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

    def range(self, start, stop):
        """Return numpy array between start and stop.

        """
        # Handle negative positions
        stop = self._get_position(stop)

        self.position = int(start)
        length = stop - start
        return self.next_window(length)

    def windows(self, length, start=None, stop=None, offset=None):

        if start is not None:
            start = self._get_position(start)
            self.position = int(start)
        if stop is None:
            stop = self._file_stop_positions[-1]

        stop = self._get_position(stop)

        while self.position + length < stop:
            yield self.next_window(length)


class Signal(object):
    """Handel FCSignal files.

    """

    def __init__(self, filename, seperator=';'):

        # Store the filelist
        self._filename = filename
        self._fileobj = open(self._filename, 'rb')

        # Set seperator of fcsignal txtfile
        self._seperator = seperator

        # Set event and position to the start of the fcsignal
        self._goto_start()

    def _goto_start(self):
        """Go back to start position in fcsignal.

        """
        self._fileobj.seek(0)
        self._next_position = 0
        self._next_event = 0

    def __getitem__(self, key):
        self.event_nr = key
        return self.next()

    @property
    def position(self):
        """Get and set the position.

        Postion will always be the next event of set position.
        """
        return self._next_position

    @position.setter
    def position(self, position):
        if position < self.position:
            self._goto_start()

        while self.position < position:
            self.next()

    @property
    def event_nr(self):
        """Get and set the event.

        """
        return self._next_event

    @event_nr.setter
    def event_nr(self, nr):
        if nr < self.event_nr:
            self._goto_start()

        while self.event_nr < nr:
            self.next()

    def __iter__(self):
        self._goto_start()
        return self

    def next(self):
        """Return next event of signal.

        """

        # Read next line from fcsignal file
        linestr = self._fileobj.readline()

        if not linestr:
            raise StopIteration

        # Split the line
        line = linestr.replace(' ', '').split(self._seperator)

        # Create a level from the line
        position = self._next_position
        state = int(line[0])
        length = int(line[1])
        value = float(line[2])
        level = [position, state, length, value]

        # Calculate the position of next level and increment event
        self._next_position += length
        self._next_event += 1

        # Return the level
        return level

    def events(self, start=0, stop=None):
        """Return event iterator.

        """

        self.event_nr = start

        while self._next_event < stop or stop is None:
            yield self.next()

    def next_events(self, nr_of_events):
        """Return number of events in length.

        """
        start = self.event_nr
        stop = start + nr_of_events

        for event in self.events(start, stop):
            yield event

    def read_events(self, start=0, stop=None):
        return list(self.events(start, stop))

    def range(self, start=0, stop=None):
        """Return interator over range (start, stop].

        """

        self.position = start

        while self._next_position < stop or stop is None:
            yield self.next()

    def next_range(self, length):
        """Return number of events in length.

        """
        start = self.position
        stop = start + length

        for event in self.range(start, stop):
            yield event

    def read_range(self, start=0, stop=None):
        return list(self.range(start, stop))


class Histogram(object):

    def __init__(self, data, bins=1000, range=None):
        self._freqs, self._bins = np.histogram(data, bins, range)

    def add(self, data):
        """Add data to the histogram.

        """
        self._freqs += np.histogram(data, self._bins)[0]

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
        return self.bins, self.freqs

    @property
    def values_n(self):
        return self.bins, self.freqs_n


class PyHistogram(object):

    def __init__(self, datapoints=[]):
        self.comment = ''
        self._elements = 0
        self._histo = defaultdict(int)

        self.add(datapoints)

    def __getitem__(self, bin):
        return self._histo[bin]

    @property
    def elements(self):
        """Return number of elements in histogram.

        """
        return self._elements

    def add_datapoint(self, bin, freq=1):
        """Add freq to bin.

        """
        self._histo[bin] += freq
        self._elements += freq

    def add(self, datapoints):

        # Try to iterate over data
        try:
            for datapoint in datapoints:
                self._histo[datapoint] += 1
                self._elements += 1
        # If not iteratable just insert
        except TypeError:
            self.add_datapoint(datapoints)

    @property
    def bins(self):
        bins = np.array(self._histo.keys())
        return bins

    @property
    def freqs(self):
        """Return frequencies.

        """
        freqs = np.array(self._histo.values())
        return freqs

    @property
    def freqs_n(self):
        """Return normed frequencies.

        """
        return self.freqs / float(self.elements)


class Times(object):

    def __init__(self):
        self._times = {}

    def __getitem__(self, key):
        return self._times[key]

    def add(self, fcsignal):

        # Iterate over all signals
        for level in fcsignal:

            state = level[0]
            length = level[1]

            try:
                histo = self._times[state]
            except KeyError:
                histo = Histogram()
                self._times[state] = histo

            # Update date time distribution dictonary
            histo.add_datapoint(length, 1)


class Fit(object):

    def __init__(self, function, xdata, ydata, start_parameters):
        self._function = function
        self._parameters, self._error = curve_fit(function, xdata, ydata,
                                                  start_parameters)

    def __call__(self, x):
        return self._function(x, *self._parameters)

    def values(self, x):
        y = self._function(x, *self._parameters)
        return x, y

    @property
    def function(self):
        return self._function

    @property
    def parameters(self):
        return self._parameters

    @property
    def error(self):
        return self._error


def flinear(x, m=1, y0=0):
    """Linear function.

    """
    x = np.array(x, copy=False)
    return m * x + y0


def fit_linear(xdata, ydata, m=1, y0=0):
    """Fit data with linear function.

    """
    return Fit(flinear, xdata, ydata, (m, y0))


def fexp(x, a=1, tau=1):
    """Exponential function.
    """
    x = np.array(x, copy=False)
    return a * np.exp(tau * x)


def fit_exp(xdata, ydata, a=1, tau=1):
    """Fit data with exponential function.

    """
    return Fit(fexp, xdata, ydata, (a, tau))


def fnormal(x, a=1, mu=0, sigma=1):
    """Normal distribution.

    """
    x = np.array(x, copy=False)
    return a * np.exp(-(x - mu)**2 / (2. * sigma**2))


def fit_normal(xdata, ydata, a=1, mu=0, sigma=1):
    """Fit data with a normal distribution.

    """
    return Fit(fnormal, xdata, ydata, (a, mu, sigma))


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


def fit_levels(data, start_parameters):
    """Create histogram from data and fit with a normal function.

    start_parameters: (a_0, mu_0 sigma_0, .... a_N, mu_N, sigma_N)
    """
    histo = Histogram(data)
    fit = Fit(flevels, histo.bins, histo.freqs_n, start_parameters)
    return fit, histo


def fit_trace(windows, start_parameters):
    fit_parameters = []

    for window in windows:
        fit, histogram = fit_levels(window, start_parameters)
        fit_parameters.append(fit.parameters)
        start_parameters = fit.parameters

    return fit_parameters


def read_fitfile(fitfile):
    """Read the fitting parameters file created by the fit_levels fucntion.

    """

    # Create list to store the fit parameters
    fit_parameters = []

    # Open fitfile
    with open(fitfile) as fobj:

        # Read all data
        for line in fobj:
            # Make a list from line
            line = line.replace('\n', '').replace(' ', '').split(',')

            # Turn all values to float
            line = [float(value) for value in line]

            # Add line to parameter list
            fit_parameters.append(line)

    # Transpose the parameters and return numpy array
    return np.transpose(fit_parameters)


def tc(sampling_rate, timestr='s'):
    """Returns the time constant based on the sampling rate and timestr.

    timestr: 'us', 'ms', 's', 'm', 'h', 'd'
    """

    sampling_rate = int(sampling_rate)

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

    # Get the number of steps out of bits
    steps = 2 ** bits - 1

    # Process the data values and return numpy array
    values = [value * (max - min) / float((steps)) + min for value in values]
    return np.array(values)


def create_histogram(data, normed=True, comment=''):
    """Create a histogram out of a data list.

    """

    # Create histogram instance
    histogram = PyHistogram()

    # Make numpy array out of data
    data = np.array(data)

    # Create histogram
    nr_of_bins = data.max() - data.min()
    elements, bins = np.histogram(data, nr_of_bins, normed=normed)

    histogram.elements = sum(elements)
    histogram.comment = comment
    histogram.data = np.array([bins[:-1], elements])
    return histogram


def read_histogram(histogram_file):

    histograms = []

    bins = []
    items = []

    with open(histogram_file, 'r') as histoobj:

        section = None

        for line in histoobj:

            # Remove trailing characters and whitespaces at line end
            line = line.rstrip()

            # Detect the section
            try:
                if (line[0] == '<' and line[-1] == '>'):
                    section = line[1:-1]

                    # Read section data
                    if section == 'fchistogram':
                        histograms.append(PyHistogram())
                        del bins[:]
                        del items[:]
                    elif section == '/fchistogram':
                        histograms[-1].data = np.array([bins, items])
                        section = None
                    elif section == '/elements':
                        section = None
                    elif section == '/comment':
                        section = None
                    elif section == '/data':
                        section = None
                    continue
            except:
                pass

            if section == 'elements':
                histograms[-1].elements = int(line)
            elif section == 'comment':
                histograms[-1].comment += line
            elif section == 'data':
                line = line.replace(' ', '').split(';')
                bins.append(float(line[0]))
                items.append(int(line[1]))

    return histograms


def read_cummulants(cummulants_file):
    """Read the cummulants from file.

    """

    data = []

    with open(cummulants_file, 'r') as cummulants_fobj:

        for line in cummulants_fobj:
            line = line.rstrip().rstrip(';')
            line = line.split(';')
            line = [float(cummulant) for cummulant in line]

            data.append(line)

    return np.transpose(data)
