cdef extern from "<complex.h>" nogil:
    double complex clog(double complex z)
    double complex csqrt(double complex z)
    double cimag(double complex z)
    double creal(double complex z)
    double complex conj(double complex z)
