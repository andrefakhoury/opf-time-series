import numpy as np
from numpy.ctypeslib import ndpointer
import numpy.ctypeslib as npct
import ctypes as ct

class OPFClassifierCInterface(ct.Structure):
	_fields_ = [
		("F_DISTANCE", ct.c_int),
		("label", ct.c_void_p),
		("orderedNodes", ct.c_void_p),
		("prototypes", ct.c_void_p),
		("cost", ct.c_void_p),
		("X", ct.c_void_p)
	]

lib = npct.load_library('opfmodule', 'src')
class OPFClassifier():
	def __init__(self, distance_method="euclidean-distance"):
		self.opf_interface = OPFClassifierCInterface()
		lib.init(ct.c_char_p(bytes(distance_method, 'utf-8')), ct.pointer(self.opf_interface))

	def fit(self, X, y):
		lib.fit(
			X.copy().reshape(-1).astype(np.double).ctypes.data_as(ct.POINTER(ct.c_double)),
			y.astype(np.int32).ctypes.data_as(ct.POINTER(ct.c_int32)),
			X.shape[0],
			X.shape[1],
			y.shape[0],
			ct.pointer(self.opf_interface)
		)

	def classify(self, X):
		preds = np.zeros(X.shape[0], dtype=np.int32)
		lib.classify(
			X.copy().reshape(-1).ctypes.data_as(ct.POINTER(ct.c_double)),
			X.shape[0],
			X.shape[1],
			preds.ctypes.data_as(ct.POINTER(ct.c_int)),
			ct.pointer(self.opf_interface)
		)
		return preds

	def get_prototypes(self):
		l = lib.getPrototypeSize(ct.pointer(self.opf_interface))
		return [lib.getPrototypeAt(i, ct.pointer(self.opf_interface)) for i in range(l)]
