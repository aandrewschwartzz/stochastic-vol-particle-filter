from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
cimport cython
import numpy as np
cimport numpy as cnp
from scipy.stats import levy_stable

@cython.boundscheck(False)
@cython.wraparound(False)
def copy_trajectory_group(cnp.ndarray[object, ndim=1] trajectories):
    cdef int n = len(trajectories)
    cdef cnp.ndarray[object, ndim=1] copied_trajectories = np.zeros(n, dtype=object)
    cdef TrajectoryList trajectory, copied_trajectory
    cdef int i

    for i in range(n):
        trajectory = <TrajectoryList>trajectories[i]
        copied_trajectory = trajectory.copy()
        copied_trajectories[i] = copied_trajectory

    return copied_trajectories

@cython.boundscheck(False)
@cython.wraparound(False)
def append_trajectory_group(cnp.ndarray[object, ndim=1] trajectories, cnp.ndarray[double, ndim=1] values):
    cdef int n = len(trajectories)
    cdef int i

    if len(values) != n:
        raise ValueError("Length of values array must match length of trajectories array")

    for i in range(n):
        trajectory = <TrajectoryList>trajectories[i]
        trajectory.append(values[i])


cdef class TrajectoryList:
    cdef double *data
    cdef size_t capacity
    cdef size_t length

    def __cinit__(self, double initial_data=0.0, size_t max_size=500):
        self.length = 1
        self.capacity = max_size
        self.data = <double*>malloc(self.capacity * sizeof(double))
        if not self.data:
            raise MemoryError("Failed to allocate memory")
        self.data[0] = initial_data
        
    def __dealloc__(self):
        if self.data is not NULL:
            free(self.data)
            self.data = NULL

    def append(self, double value):
        if self.length >= self.capacity:
            raise OverflowError("Appending would exceed allocated capacity")
        self.data[self.length] = value
        self.length += 1

    def __getitem__(self, size_t index):
        if index >= self.length or index < 0:
            raise IndexError("Index out of bounds")
        return self.data[index]

    def __setitem__(self, size_t index, double value):
        if index >= self.length or index < 0:
            raise IndexError("Index out of bounds")
        self.data[index] = value

    def __len__(self):
        return int(self.length)
        
        
    def copy(self):
        cdef TrajectoryList copied = TrajectoryList(max_size=self.capacity)
        copied.length = self.length
        memcpy(copied.data, self.data, sizeof(double) * self.length)
        return copied
        
    def to_list(self):
        return [self.data[i] for i in range(self.length)]
    
    cdef TrajectoryList _slice(self, Py_ssize_t start, Py_ssize_t stop):
        cdef TrajectoryList sliced = TrajectoryList(max_size=stop-start)
        sliced.length = stop - start
        memcpy(sliced.data, self.data + start, sizeof(double) * (stop - start))
        return sliced

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.length)
            if step != 1:
                raise IndexError("Slice step must be 1")
            if start < 0 or stop > self.length or start > stop:
                raise IndexError("Slice indices out of bounds")
            return self._slice(start, stop)
        else:
            if index >= self.length or index < 0:
                raise IndexError("Index out of bounds")
            return self.data[index]
        

'------------------------------------------------------------------------------------'

