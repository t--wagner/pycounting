# -*- coding: utf-8 -*-

from collections import defaultdict
import numpy as np


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