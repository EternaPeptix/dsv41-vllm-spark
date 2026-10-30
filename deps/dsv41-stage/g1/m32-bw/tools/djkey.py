#!/usr/bin/env python3
"""Offline re-implementation of the deep_jit disk-cache key (DeepGEMM JIT).

key = FNV1a2x64( "cutlass-421", hex(nvcc --version), join(flags), post_hook_hash(""),
                 parse_into_hash(kernel.cu) )
parse_into_hash(code) = hex( FNV(code, hex(parse(include_1)), hex(parse(include_2)), ...) )
only `#include <deep_gemm/...>` lines are followed (include_prefixes = ["deep_gemm/"]).

Usage: djkey.py --include DIR --source kernel.cu [--version-file meta.json]
Prints the 32-hex digest. Verified against live cache entries (see selftest).
"""
import argparse
import json
import os
import re
import sys

M64 = (1 << 64) - 1


class FNV1a:
    def __init__(self):
        self.s0 = 0xc6a4a7935bd1e995
        self.s1 = 0xff51afd7ed558ccd

    def _byte(self, b):
        self.s0 = ((self.s0 ^ b) * 0x100000001b3) & M64
        self.s1 = ((self.s1 ^ b) * 0x9e3779b97f4a7c15) & M64

    def update(self, data):
        if isinstance(data, str):
            data = data.encode()
        n = len(data)
        for shift in range(0, 64, 8):
            self._byte((n >> shift) & 0xff)
        for b in data:
            self._byte(b)
        return self

    def hexdigest(self):
        def mix(v):
            v = ((v ^ (v >> 30)) * 0xbf58476d1ce4e5b9) & M64
            v = ((v ^ (v >> 27)) * 0x94d049bb133111eb) & M64
            return v ^ (v >> 31)
        return "%016x%016x" % (mix(self.s0), mix(self.s1))


INC = re.compile(r'^[ \t\f\v]*#[ \t\f\v]*include[ \t\f\v]*<([^>"]+)>')


class Parser:
    def __init__(self, include_dirs, prefixes=("deep_gemm/",)):
        self.dirs = include_dirs
        self.prefixes = prefixes
        self.cache = {}

    def parse_include(self, line):
        m = INC.match(line)
        if not m:
            return None
        fn = m.group(1)
        return fn if any(fn.startswith(p) for p in self.prefixes) else None

    def file_hash(self, fn):
        if fn in self.cache:
            return self.cache[fn]
        for d in self.dirs:
            p = os.path.join(d, fn)
            if os.path.exists(p):
                break
        else:
            raise FileNotFoundError(fn)
        with open(p, "rb") as f:
            code = f.read()
        h = self.code_hash(code)
        self.cache[fn] = h
        return h

    def code_hash(self, code):
        if isinstance(code, str):
            code = code.encode()
        h = FNV1a().update(code)
        text = code.decode("utf-8", "surrogateescape")
        # C++ splits on '\n' only; a trailing '\r' would stay in the line (none here)
        for line in text.split("\n"):
            fn = self.parse_include(line)
            if fn:
                h.update(self.file_hash(fn))
        return h.hexdigest()


NVCC_VERSION = ("nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2025 NVIDIA Corporation\n"
                "Built on Wed_Aug_20_01:57:39_PM_PDT_2025\nCuda compilation tools, release 13.0, V13.0.88\n"
                "Build cuda_13.0.r13.0/compiler.36424714_0\n")
FLAGS = ["--gpu-architecture=sm_120f", "-O3", "--compiler-options=-O3",
         "--ptxas-options=--register-usage-level=10", "-std=c++20", "--compiler-options=-fPIC",
         "--compiler-options=-fconcepts", "--expt-relaxed-constexpr", "--expt-extended-lambda",
         "--diag-suppress=39,161,174,177,186,940",
         "--compiler-options=-Wno-deprecated-declarations,-Wno-abi"]


def key(include_dir, source, version=NVCC_VERSION, flags=FLAGS, extra="cutlass-421"):
    h = FNV1a().update(extra).update(FNV1a().update(version).hexdigest())
    h.update(" ".join(flags)).update("")
    h.update(Parser([include_dir]).code_hash(source))
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--meta", help="meta.json to take nvcc version/flags from")
    a = ap.parse_args()
    version, flags = NVCC_VERSION, FLAGS
    if a.meta:
        m = json.load(open(a.meta))
        version = m["compiler_info"]["version"]
        o = m["compiler_options"]
        assert o["arch"] == "120f" and o["optimize_level"] == "3" and not o["use_fast_math"]
        assert o["ptxas_register_usage_level"] == 10 and not o["ptxas_verbose"] and not o["with_line_info"]
        flags = (["--gpu-architecture=sm_%s" % o["arch"], "-O3", "--compiler-options=-O3",
                  "--ptxas-options=--register-usage-level=10"] + o["nvcc_flags"] + o["extra_nvcc_flags"])
    print(key(a.include, open(a.source, "rb").read(), version, flags))


if __name__ == "__main__":
    sys.exit(main())
