%%cython
import numpy as np
cimport numpy as np

def detect(np.ndarray[np.int, ndim=1] trace,
           double limit0_down, double limit0_up,
           double limit1_down, double limit1_up,
           signal):

    # Datapoint variables
    cdef long datapoint
    cdef int  datapoint_state

    # Level variables
    cdef int  level_state
    cdef long level_length
    cdef long level_value

    level_state  = signal[-1][0]
    level_length = signal[-1][1]
    level_value  = signal[-1][2]

    # Iterate through array by c stlye indexing
    cdef unsigned long i
    for i in range(trace.shape[0]):

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