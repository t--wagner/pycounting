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


class BiTrace(object):
    """Simulate a counting Trace.

    """

    def __init__(self, rate0, rate1, level0=0, level1=1, sampling_rate=100e3,
                 noise_sigma=0, delay=1, fosc=None):

        # Raten für das Level
        self._rates = (rate0, rate1)
        self._level0 = level0
        self._level1 = level1
        self._delta = level1 - level0
        self._sampling_rate = sampling_rate
        self._fosc = fosc

        # Noise
        self._noise_sigma = noise_sigma

        # Delay
        self._delay = delay
        self._delay_stepsize = self._delta / float(self._delay)
        self._delay_stepnr = 0

        # Function on top
        #self._function = function

        # Prepare everything
        self._state = 0

    def __iter__(self):
        return self

    def next(self):

        # Get rate for current state
        rate = self._rates[self._state]

        #Roll the Dice and toogle state
        if rate / float(self._sampling_rate) > random.random():
            self._state = 0 if self._state else 1

        # Delay simulation
        if self._state == 0 and self._delay_stepnr > 0:
            self._delay_stepnr -= 1
        elif self._state == 1 and self._delay_stepnr < self._delay:
            self._delay_stepnr += 1


        noise = random.gauss(0, self._noise_sigma)
        delay = self._delay_stepnr * self._delay_stepsize

        if self._state == 1:
            binary = self._level1
            delay = delay - self._delta
        else:
            binary = self._level0

        if self._fosc:
            osc = next(self._fosc)
        else:
            osc = 0

        # Return tuple with all trace components
        return binary, noise, delay, osc

    def range(self, length=None):

        if not length:
            length = self._sampling_rate

        return [self.next() for i in xrange(length)]


class BiSignal(object):

    def __init__(self, rate0, rate1, state=0, position=0, sampling_rate=100e3):

        self._rate = [rate0, rate1]
        self._sampling_rate = sampling_rate
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

            p = rate / float(self._sampling_rate)

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