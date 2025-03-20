import ctypes
import numpy as np

class CNLA():
    def __init__(self):

        lib_cnla = ctypes.CDLL('./cnla/lib_cnla.so')

        ND_POINTER = np.ctypeslib.ndpointer(dtype=np.float64,
                ndim=2,
                flags="C")

        lib_cnla.QR.argtypes = [ND_POINTER, ctypes.c_size_t]

        self.lib = lib_cnla

    def QR(self, A):
        shape = A.shape[0] * A.shape[0]
        self.lib.QR(A, *A.shape)

a = np.eye(10)

cnla = CNLA()
cnla.QR(a)

print(a)
