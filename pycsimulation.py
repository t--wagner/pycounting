# -*- coding: utf-8 -*-

import random
import numpy as np

from itertools import cycle


class ROsc(object):

    def __init__(self, rate0, amplitude, frequency, phase=0):
        self._rate0 = rate0
        self._amp = amplitude
        self._freq = frequency
        self._per = 1 / self._freq
        self._pha = phase

        x = np.linspace(0, self._per, self._per)
        rate = self._rate0 + self._amp * np.sin(self._pha + 2 * np.pi * x * self._freq)
        self._c = cycle(rate)

    def next(self):
        return self._c.next()

    def __iter__(self):
        return self


class BiSignal(object):

    def __init__(self, rate0, rate1, state=0, position=0, sample_rate=100e3):

        self._rate = [rate0, rate1]
        self._sample_rate = sample_rate
        self._state = state
        self._position = position
        self._length = 1

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position = position

    def __iter__(self):

        while True:
            rate = self._rate[self._state]

            # Wahrscheinlichkeit ändern falls keine Kon
            try:
                rate = rate.next()
            except AttributeError:
                pass

            p = rate / float(self._sample_rate)

            # Dice until True
            while p < random.random():
                self._length += 1

            # Return the level
            state = self._state
            if self._state:
                self._state = 0
            else:
                self._state = 1

            position = self._position
            self._position += self._length

            length = self._length
            self._length = 1

            yield [position, state, length, state]

    def events(self, nr, start_position=None):

        if start_position is not None:
            self._position = start_position

        event_generator = self.__iter__()

        for event_nr in xrange(nr):
            yield event_generator.next()

    def range(self, length, start_position=None):

        if start_position is not None:
            self._position = start_position

        stop = self._position + length

        for event in self.__iter__():
            yield event

            if stop <= event[0] + event[2]:
                break

    def read_events(self):
        pass

    def read_range(self):
        pass