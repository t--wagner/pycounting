# -*- coding: utf-8 -*-

from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
import glob
import pycounting as pyc
from os.path import getsize


def list_filenames(pathname):
    """List of sorted filenames in path, based on pattern.

    """
    files = glob.glob(pathname)
    return sorted(files)


class BinaryTrace(object):
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
            while self.position + length <= stop:
                yield self.next(length)

    def plot(self, start, stop, step=1, ax=None, **kwargs):

        if not ax:
            ax = plt.gca()

        x = np.arange(start, stop, step)
        y = self.__getitem__(slice(start, stop, step))

        return ax.plot(x, y, **kwargs)


def binary_to_hdf(trace, hdf_file, dataset_string, sampling_rate,
                  bit, wlength=100e3):

    wlength = int(wlength)
    dset = hdf_file.create_dataset(dataset_string, dtype=trace._dtype,
                                   shape=(len(trace),))
    hdf = pyc.Trace(dset)
    hdf.sampling_rate = sampling_rate
    hdf.bit = bit

    # Copy binary trace to HDF5 dataset
    position = 0
    for window in trace.windows(wlength, start=0, stop=len(trace)):
        hdf[position:position+wlength] = window
        position += wlength

    return hdf


class Signal(object):

    dlevel = np.dtype([('state', np.int16),
                       ('length', np.int64),
                       ('value', np.float64)])

    def __init__(self, data=None, start=0):

        # Define signals numpy datatype
        if isinstance(data, str):
            data = np.fromfile(data, dtype=self.dlevel)
        elif data is None:
            data = [(-1, 0, 0)]

        self._data = np.array(data, dtype=self.dlevel)
        self.start = start

    @classmethod
    def from_file(cls, filename):
        data = np.fromfile(filename, dtype=cls.dlevel)
        return cls(data)

    def __repr__(self):
        return repr(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __len__(self):
        return self._data.size

    @property
    def data(self):
        return self._data

    def append(self, levels):
        """Append new list of levels to signal.

        """
        new_data = np.array(levels, dtype=self.dlevel)
        self._data = np.concatenate((self._data, new_data))

    def range(self, start, stop):
        index = (self.position >= start) & (self.position < stop)
        return self.position[index], self._data[index]

    @property
    def position(self):
        """Position array of signal.

        """
        return self.start + np.cumsum(self._data['length'])

    @property
    def state(self):
        """State array of signal.

        """
        return self._data['state']

    @property
    def length(self):
        """Length array of signal.

        """
        return self._data['length']

    @property
    def value(self):
        """Value array of signal.

        """
        return self._data['value']

    @property
    def mean(self):
        """Mean value of signal.
        """
        return self._data['value'].mean()

    def save(self, filename):
        """Write everything to file.

        """
        self._data.tofile(filename)

    def flush(self, fobj=None, nr=-1):
        """Flush nr of levels to file.

        """
        if fobj:
            self._data[:nr].tofile(fobj)
        self._data = np.array(self._data[nr:], dtype=self.dlevel)

    def plot(self, start, stop, ax=None, **kwargs):

        if not ax:
            ax = plt.gca()

        x, y = self.range(start, stop)
        return ax.step(x, y['value'], **kwargs)


class SignalStream(object):
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
        state = int(line[0])
        length = int(line[1])
        value = float(line[2])
        level = [state, length, value]

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