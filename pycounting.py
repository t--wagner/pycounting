# -*- coding: utf-8 -*-

import os
import struct
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class BinaryData(object):

    def __init__(self, filelist, struct_format):

        # Store the filelist
        self._filelist = filelist

        # Open all datafiles in readable binary mode
        self._fileobjs = []
        for filename in self._filelist:
            self._fileobjs.append(open(filename, 'rb'))

        # Set position to first file
        self._fileobj = self._fileobjs[0]

        # Store the struct information
        self._struct_format = struct_format
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

        return np.array(data)


def fit_levels(fctrace, window, start_values, fitfilename, offsets=None,
               printing=True, plotting=False, timing=False, exceptions=True):

    # Create offsets from window length and fctrace
    if not offsets:
        offsets = range(0, fctrace.points, window)

    nr_of_offsets = len(offsets)
    window = int(window)

    # --- Fitting --- #
    fitfile = open(fitfilename, 'w')

    fit_values = np.array(start_values)
    data = []

    if plotting:
        xsubplts = len(offsets)
        ysubplts = 2
        fig = plt.figure(figsize=(15, 3 * xsubplts))

    # Cycle through the data
    for nr, offset in enumerate(offsets, 1):

        # Starting time
        tstart = time.time()

        # Read data in window
        data = np.array(fctrace.read(window, offset))

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

        # Plot the fit if requested
        if plotting:
            #Plot Data
            ax = fig.add_subplot(xsubplts, ysubplts, 2 * nr - 1)
            ax.set_title('t = ' + str(offset))
            xdata = np.array(range(offset, offset + plotting))
            ydata = np.array(data[:plotting])
            ax.plot(xdata, ydata)
            ax.set_xlabel('t (ms)')
            ax.set_ylabel('I (nA)')

            #Plot Histogram
            ax = fig.add_subplot(xsubplts, ysubplts, 2 * nr)
            ax.set_title('t = ' + str(offset))
            xdata = histogram[0]
            ydata = histogram[1][:-1]
            ax.plot(xdata, ydata)
            ax.plot(normal(x, *fit_values), x)

    # Optimize subplot spaceing
    if plotting:
        fig.tight_layout()

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
    """Returns the time constant based on the sampling rate.

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


class Histogram(object):

    def __init__(self):
        self.elements = 0
        self.comment = ''
        self.data = np.array([])

    def __getitem__(self, key):
        return self.data[key]

    def __repr__(self):
        return self.data.__repr__()

    def bins(self):
        return self.data[0]

    def frequencies(self):
        return self.data[1]


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

            line = line.replace('\n', '')

            # Detect and set section
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


def read_fcsignal(fcsignal_file, start_position=0, steps=-1):
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

            if counter < start_position:
                continue

            level.append(int(line[0]))
            position.append(counter)
            length.append(int(line[1]))
            value.append(float(line[2]))

            steps -= 1
            if not steps:
                break

    return [level, position, length, value]


def read_cummulants(cummulants_file):
    """Read the cummulants from file.

    """

    data = []

    with open(cummulants_file, 'r') as cummulants_fobj:

        for line in cummulants_fobj:
            line = line.replace(' ', '').replace(';\n', '')
            line = line.split(';')
            line = [float(cummulant) for cummulant in line]

            data.append(line)

    return np.transpose(data)
