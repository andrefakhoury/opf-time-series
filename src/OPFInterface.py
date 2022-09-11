import numpy as np
import numpy.ctypeslib as npct
import ctypes as ct

# Import module
mymodule = npct.load_library('opfmodule', '.')

class OPFClassifierCInterface(ct.Structure):
	_fields_ = [("x", ct.c_int)]
	def __init__(self):
		pass

def call_multiply():
	val = 20
	opfInterface = OPFClassifierCInterface()
	mymodule.init(ct.pointer(opfInterface))
	xx = mymodule.multiply(ct.pointer(opfInterface))
	print(opfInterface.x)
	
call_multiply()