# -*- coding: utf-8 -*-

import os
import struct
import time
import numpy as np
from scipy.optimize import curve_fit


class BinaryData(object):

    def __init__(self, filelist, datatype, byte_order=None):

        # Store the filelist
        self._filelist = filelist

        # Open all datafiles in readable binary mode
        self._fileobjs = []
        for filename in self._filelist:
            self._fileobjs.append(open(filename, 'rb'))

        # Set position to first file
        self._fileobj = self._fileobjs[0]

        # Store the struct information
        if datatype == 'int':
            self._struct_format = 'i'
        elif datatype == 'ushort':
            self._struct_format = 'H'
        else:
            self._struct_format = datatype

        if byte_order:
            self._struct_format = byte_order + self._struct_format

        self._datapointsize = struct.calcsize(self._struct_format)

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

    def __del__(self):
        """Close all datafiles.

        """
        for fileobj in self._fileobjs:
            fileobj.close()

    @property
    def filelist(self):
        """Return list of filenames.

        """
        return self._filelist

    @property
    def nr_of_files(self):
        """Return the number of added files.

        """
        return len(self._filelist)

    @property
    def datasize(self):
        """Return the size of all files in kBytes.

        """
        return self._datasize

    @property
    def struct_format(self):
        """Return the struct format.

        """
        return self._struct_format

    @property
    def datapointsize(self):
        """Return the size of one datapoint in kBytes.

        """
        return self._datapointsize

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

    @position.setter
    def position(self, position):
        """Set the current datapoint position.

        """

        # If negative position calculate the corresponding positive position
        if position < 0:
            position += self._file_stop_positions[-1]

            # Raise IndexError if out of range
            if position < 0:
                raise IndexError('position out of range')

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

    def read(self, points=-1, position=None):
        """Read number of points after position and return them as a list.

        If position is None the reading starts at the current position.

        """

        # Set the position if not None
        if not position is None:
            self.position = long(position)

        start  = self.position

        # Creat a list to store the data
        data = []

        # Read the points
        while(points):
            try:
                datastring = self._fileobj.read(self._datapointsize)
                data += struct.unpack(self._struct_format, datastring)
                points -= 1
            except KeyboardInterrupt:
                raise
            except:
                file_index = self._fileobjs.index(self._fileobj)
                try:
                    self._fileobj = self._fileobjs[file_index + 1]
                    self._fileobj.seek(0)
                except IndexError:
                    break

        stop = self.position

        return np.array(data), (start, stop)


def fit_levels(fctrace, window, start_values, fitfilename, offsets=None,
               printing=True, timing=False, exceptions=True):

    # Create offsets from window length and fctrace
    if not offsets:
        offsets = range(0, fctrace.points, window)

    nr_of_offsets = len(offsets)
    window = int(window)

    # --- Fitting --- #
    fitfile = open(fitfilename, 'w')

    fit_values = np.array(start_values)
    data = []


    # Cycle through the data
    for nr, offset in enumerate(offsets, 1):

        # Starting time
        tstart = time.time()

        # Read data in window
        data, data_range = fctrace.read(window, offset)

        # Create histogram
        bins = data.max() - data.min()
        histogram = np.histogram(data, bins, normed=True)

        # Perform the fit
        x = histogram[1][:-1]
        y = histogram[0]

        try:
            fit_values, b = curve_fit(normal, x, y, fit_values)
        except RuntimeError:
            if exceptions:
                raise RuntimeError

        # Save the fit paras
        para_str = str(offset) + ', ' + str(fit_values.tolist())[1:-1]
        fitfile.write(para_str + '\n')
        fitfile.flush()

        # Print the number of remaing fits
        if printing:
            print nr_of_offsets - nr,

        # Print the time for one fit
        if timing:
            print time.time() - tstart,

    #Close fit file
    fitfile.close()


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


class Histogram(object):

    def __init__(self):
        self.elements = 0
        self.comment = ''
        self._data = np.array([])

    def __getitem__(self, key):
        return self._data[key]

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def bins(self):
        return self._data[0]

    @property
    def freqs(self):
        return self._data[1]


def create_histogram(data, normed=True, comment=''):
    """Create a histogram out of a data list.

    """

    # Create histogram instance
    histogram = Histogram()

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
                        histograms.append(Histogram())
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


def linear(xdata, slope=1, yintercept=0):

    # Make numpy array out of data
    xdata = np.array(xdata)

    return slope * xdata + yintercept


def fit_linear(xdata, ydata, slope=1, yintercept=0, function=True):

    ydata = np.array(ydata)
    start_paras = np.array([slope, yintercept])

    fit = curve_fit(linear, xdata, ydata, start_paras)

    if function:
        return linear(xdata, *fit[0])
    else:
        return fit


def exp(xdata, a=1, tau=1):

    # Make numpy array out of data
    xdata = np.array(xdata)

    return a * np.exp(tau * xdata)


def fit_exp(xdata, ydata, a=1, tau=1, function=True):

    xdata = np.array(xdata)
    ydata = np.array(ydata)
    start_paras = np.array([a, tau])

    fit = curve_fit(exp, xdata, ydata, start_paras)

    if function:
        return exp(xdata, *fit[0])
    else:
        return fit


# Multiple Gauss functions added
def normal(x, *parameters):
    """Create a gauss sum function from on the parameter list.

    """

    # Define gauss function
    def normal(x, sigma, mu, A):
        x = np.array(x)
        return A * np.exp(-(x - mu)**2 / (2. * sigma**2))

    x = np.array(x)
    res = np.zeros(x.size)

    pars_list = [parameters[i:i+3] for i in range(0, len(parameters), 3)]

    for pars in pars_list:
        res += normal(x, *pars)

    return res


def fit_normal(xdata, ydata, function=True, *parameters):

        # Make numpy arrays out of data
        xdata = np.array(xdata)
        ydata = np.array(ydata)

        start_paras = np.array(parameters)

        fit = curve_fit(normal, xdata, ydata, start_paras)

        if function:
            return normal(xdata, *fit[0])
        else:
            return fit


class FCSignal(object):

    def __init__(self, *filenames):
        self._filenames = filenames

    def filenames(self):
        return self._filenames

    def __iter__(self):
        pass

    def read(self, start=None, stop=None, steps=None):
        pass

    def postion(self):
        pass


def read_fcsignal(fcsignal_file, start=0, stop=None, steps=-1):
    """Read data from fcsignal.

    """

    level = []
    position = []
    length = []
    value = []

    counter = 0

    with open(fcsignal_file, 'r') as fcsignal_fobj:

        #Cut file header
        fcsignal_fobj.readline()

        #Read Data
        for line in fcsignal_fobj:
            line = line.replace(' ', '').split(';')
            counter += int(line[1])

            # Check start position
            if counter < start:
                continue

            # Check stop position
            if stop:
                if position > stop:
                    break

            level.append(int(line[0]))
            position.append(counter)
            length.append(int(line[1]))
            value.append(float(line[2]))

            steps -= 1
            if not steps:
                break

    return [level, position, length, value]


def time_distribution(fcsignal, start=0, stop=None, steps=-1, seperator=';'):
    """Extract time distributions from the fcsignal.

    """

    # Define start parameters
    start = int(start)
    stop  = int(stop)
    steps = int(steps)
    position = 0

    # Create level directory to store time distributions
    times = {}

    with open(fcsignal, 'r') as fcsignal_fobj:

        #Cut file header
        fcsignal_fobj.readline()

        #Read Data
        for line in fcsignal_fobj:

            # Split signal line
            line = line.rstrip()
            line = line.replace(' ', '').split(seperator)

            # Get level values
            level  = int(line[0])
            position += int(line[1])
            length = int(line[1])

            # Check start position
            if position < start:
                continue

            # Check stop position
            if stop:
                if position > stop:
                    break

            # Get the time distribution from level dictonary or create it
            try:
                histo = times[level]
            except KeyError:
                histo = {}
                times[level] = histo

            # Update date time distribution dictonary
            try:
                freq = histo[length]
                histo[length] = freq + 1
            except KeyError:
                histo[length] = 1

            steps -= 1
            if not steps:
                break

    #Format the dictonaries
    for level, histo in times.items():
        histogram = Histogram()
        histogram.data = np.array([histo.keys(), histo.values()])
        times[level] = histogram

    return times


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



