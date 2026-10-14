"""Current Darwin RSS via documented local SDK field, not ru_maxrss unit guesses."""
import ctypes,os,sys

def resident_bytes(pid):
    if sys.platform!='darwin':return None
    lib=ctypes.CDLL('/usr/lib/libproc.dylib')
    f=lib.proc_pidinfo;f.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_uint64,ctypes.c_void_p,ctypes.c_int];f.restype=ctypes.c_int
    # SDK proc_taskinfo starts with virtual size and resident size, both uint64 bytes.
    b=ctypes.create_string_buffer(128);n=f(pid,4,0,b,len(b))
    if n<16:return None
    return int.from_bytes(b.raw[8:16],sys.byteorder)
