%%cython
import numpy as np
cimport numpy as np


# Declare fused type to use are generic datatype
ctypedef fused datatype:
    short
    int
    long
    float
    double


# Take 1d numpy arrays
def digitize(np.ndarray[datatype, ndim=1] trace,
             signal,
             int average,
             double center_0, double width_0,
             double center_1, double width_1):

    #Calculate levels
    cdef double limit0_down = center_0 - width_0
    cdef double limit0_up = center_0 + width_0
    cdef double limit1_down = center_1 - width_1
    cdef double limit1_up = center_1 + width_1

    # Datapoint variables
    cdef datatype datapoint
    cdef int datapoint_state

    # Level variables
    cdef int  level_state
    cdef long level_length
    cdef datatype level_value

    level_state, level_length, level_value = signal[-1]

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
            level_value += (datapoint - level_value) / level_length;
        elif datapoint_state == -1:
            # Current state is undefined
            pass
        else:
            # State changed
            signal.append((level_state, level_length, level_value))

            # Reset level
            level_state  = datapoint_state
            level_length = 0
            level_value  = datapoint

    # Return the buffer that is necassary to buffer
    return trace[-average:].copy()