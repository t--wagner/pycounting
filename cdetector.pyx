import numpy as np
cimport numpy as np

# Declare fused type to use are generic datatype
ctypedef fused datatype:
    short
    int
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
    cdef int  level_state
    cdef long level_length
    cdef double level_value

    # Get the values from the last level as starting position
    level_state, level_length, level_value = signal[-1]
    del signal[-1]

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
            signal.append((level_state, level_length, level_value))

            # Reset level
            level_state  = datapoint_state
            level_length = 0
            level_value  = datapoint

    #Append the unfinished levels
    signal.append((level_state, level_length, level_value))

    # Return the buffer that is necassary to buffer
    return trace[-average:].copy()