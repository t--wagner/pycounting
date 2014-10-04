# -*- coding: utf-8 -*-

from collections import defaultdict
import numpy as np


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