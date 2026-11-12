"""(copied verbatim from climb-r5/harness/pdl/r1-pdl-devtest-v21/tests/device/graph_edges.py for r6-comm device_test.py)
Programmatic-edge inspection of a captured CUDA graph (driver API via ctypes), after spark-ds41f 0522
(tests/kernels/test_deepseek_v41_tilelang_pdl.py: _EdgeData / _KernelNodeParams, PROGRAMMATIC = 1).

A kernel launched with cudaLaunchAttributeProgrammaticStreamSerialization during stream capture gets its incoming
kernel->kernel dependency as a CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC edge. If the attribute did not reach the driver
(e.g. Triton dropped launch_pdl, or the launch went through a path without it) the edge is DEFAULT, the kernel is
fully serialised and a bitwise test proves nothing about the wait. So the race test asserts the edge type.

Symbols checked present in /lib/aarch64-linux-gnu/libcuda.so.1 on spark-one (driver 580.178.04, 2026-11-08):
cuGraphGetNodes, cuGraphGetEdges_v2, cuGraphNodeGetType, cuGraphKernelNodeGetParams_v2, cuFuncGetName, cuKernelGetName.
"""
import ctypes

PROGRAMMATIC = 1          # CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC
KERNEL_NODE = 0           # CU_GRAPH_NODE_TYPE_KERNEL


class _EdgeData(ctypes.Structure):
    _fields_ = [("from_port", ctypes.c_ubyte), ("to_port", ctypes.c_ubyte), ("type", ctypes.c_ubyte),
                ("reserved", ctypes.c_ubyte * 5)]


class _KernelNodeParams(ctypes.Structure):          # CUDA_KERNEL_NODE_PARAMS_v2
    _fields_ = [("func", ctypes.c_void_p), ("grid", ctypes.c_uint * 3), ("block", ctypes.c_uint * 3),
                ("shared_mem", ctypes.c_uint), ("params", ctypes.c_void_p), ("extra", ctypes.c_void_p),
                ("kern", ctypes.c_void_p), ("ctx", ctypes.c_void_p)]


_lib = None


def _cu():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL("libcuda.so.1")
        for n in ("cuGraphGetNodes", "cuGraphGetEdges_v2", "cuGraphNodeGetType", "cuGraphKernelNodeGetParams_v2",
                  "cuFuncGetName", "cuKernelGetName"):
            getattr(_lib, n).restype = ctypes.c_int
    return _lib


def _ck(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} failed: CUresult {rc}")


def _name(node):
    cu = _cu()
    t = ctypes.c_int()
    _ck(cu.cuGraphNodeGetType(ctypes.c_void_p(node), ctypes.byref(t)), "cuGraphNodeGetType")
    if t.value != KERNEL_NODE:
        return None, f"<node type {t.value}>"
    p = _KernelNodeParams()
    _ck(cu.cuGraphKernelNodeGetParams_v2(ctypes.c_void_p(node), ctypes.byref(p)), "cuGraphKernelNodeGetParams_v2")
    s = ctypes.c_char_p()
    if p.func:
        _ck(cu.cuFuncGetName(ctypes.byref(s), ctypes.c_void_p(p.func)), "cuFuncGetName")
    elif p.kern:
        _ck(cu.cuKernelGetName(ctypes.byref(s), ctypes.c_void_p(p.kern)), "cuKernelGetName")
    else:
        return True, "<kernel ?>"
    return True, (s.value or b"").decode(errors="replace")


def edges(raw_graph):
    """[(src_name, src_is_kernel, dst_name, dst_is_kernel, edge_type, from_port, to_port)] for a cudaGraph_t."""
    cu = _cu()
    g = ctypes.c_void_p(raw_graph)
    n = ctypes.c_size_t(0)
    _ck(cu.cuGraphGetEdges_v2(g, None, None, None, ctypes.byref(n)), "cuGraphGetEdges_v2(count)")
    k = n.value
    if k == 0:
        return []
    src, dst = (ctypes.c_void_p * k)(), (ctypes.c_void_p * k)()
    data = (_EdgeData * k)()
    _ck(cu.cuGraphGetEdges_v2(g, src, dst, data, ctypes.byref(n)), "cuGraphGetEdges_v2")
    cache, out = {}, []
    for i in range(n.value):
        for h in (src[i], dst[i]):
            if h not in cache:
                cache[h] = _name(h)
        sk, sn = cache[src[i]]
        dk, dn = cache[dst[i]]
        out.append((sn, bool(sk), dn, bool(dk), data[i].type, data[i].from_port, data[i].to_port))
    return out


def incoming(edge_list, name_pred):
    """Edges whose destination kernel name satisfies name_pred."""
    return [e for e in edge_list if e[3] and name_pred(e[2])]
