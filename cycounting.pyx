import numpy as np
cimport numpy as np

# Declare fused type to use are generic datatype
ctypedef fused datatype:
    unsigned short
    short
    unsigned int
    int
    unsigned long
    long
    float
    double


def digitize(np.ndarray[datatype, ndim=1] trace,
             signal,
             int average,
             double limit0_down, double limit0_up,
             double limit1_down, double limit1_up):

    # Datapoint variables
    cdef datatype datapoint
    cdef int datapoint_state

    # Level variables
    new_signal = list()
    cdef int  level_state
    cdef long level_length
    cdef double level_value

    # Get the values from the last level as starting position
    first_level = signal[-1]
    level_state, level_length, level_value = first_level

    # Iterate through array by c stlye indexing. Keep always enough points for averraging.
    cdef unsigned long i
    for i in range(trace.shape[0] - average):

        # Get the next datapoint and increase level length
        datapoint = trace[i]
        level_length += 1

        # Get the datapoint state
        if limit0_down < datapoint <= limit0_up:
            datapoint_state = 0
        elif limit1_down < datapoint <= limit1_up:
            datapoint_state = 1
        else:
            datapoint_state = -1

        # Compare current and last state
        if datapoint_state == level_state:
            # State did not change
            level_value = (1 - 1 / <float>level_length) * level_value + datapoint / <float>level_length
        elif datapoint_state == -1:
            # Current state is undefined
            pass
        else:
            # State changed
            # We us append here becaus new_signal is here a list
            new_signal.append((level_state, level_length, level_value))

            # Reset level
            level_state  = datapoint_state
            level_length = 0
            level_value  = datapoint

    # Extend unfinished stuff
    new_signal.append((level_state, level_length, level_value))

    # Update last level and extend the rest
    signal[-1] = new_signal[0]
    #del new_signal[0]
    signal.extend(new_signal[1:])

    # Return the buffer that is necassary to buffer
    new_signal[0] = (new_signal[0][0], new_signal[0][1] - first_level[1], new_signal[0][2])
    return new_signal, trace[-average:].copy()


def count(np.ndarray[datatype, ndim=1] events,
          datatype delta,
          datatype offset,
          unsigned long counts,
          histogram):

    cdef datatype event

    for event in events:

        while ((offset + delta) < event):

            histogram[counts] += 1
            counts = 0
            offset += delta

        counts += 1

    return offset, counts
