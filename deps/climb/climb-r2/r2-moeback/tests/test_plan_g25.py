#!/usr/bin/env python3
"""G25 plumbing test on CPU: the real MoERunner / SharedExperts / Athena gather code (base and patched), driven
with stubbed vllm imports, CPU tensors, emulated kernels and logged streams/events/torch ops.

What is real: MoERunner (whole class, exec'd from the live base file and from the kit's patched render),
SharedExperts (whole class from the live file), gated_gather() (from the live / patched
fc2_status_gated_gather_trial.py; test-only transform: the is_cuda device check is relaxed for CPU tensors),
and the kit's dsv41_moe_fold.py. What is emulated: Triton kernel launches (_gated_gather, _gated_gather_add,
CPU emulation with explicit fp32 accumulation and bf16 RNE stores), the shared MLP, the routed FC1/FC2
(a deterministic bf16 mm2_out), events/streams (logged), the all-reduce (records its input).

Checks per scenario:
  flag OFF (patched)  == base: identical op/stream/event log, identical AR-input bits.
  flag ON             : identical AR-input bits; log == base log except gather+add -> wait(shared ready
                        event on main) + gather_add (no add, one fewer allocation); the shared output is
                        consumed exactly once (SharedExperts._output is None afterwards).
Scenarios: overlap (aux stream), no-overlap (M above the stream threshold), drafter/non-Athena experts,
routed_scaling_factor != 1, fused-output pre-reduced, padded routed width (declined), status failed,
double gather under one MoE call (must raise).
Run: ~/exo-venv313/bin/python3 test_plan_g25.py <kit_dir>
"""
import contextlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import subprocess
import sys
import tempfile
import types
from enum import IntEnum
from types import SimpleNamespace

import torch

KIT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".."))
LOG = []
STREAM = ["main"]


# ------------------------------------------------------------------ stubs
class Dummy:
    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        return Dummy()

    def __getattr__(self, name):
        return Dummy()


def dummy_cls(name):
    return type(name, (Dummy,), {})


class StubModule(types.ModuleType):
    def __init__(self, name, overrides):
        super().__init__(name)
        self.__path__ = []
        self._ov = overrides

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if f"{self.__name__}.{name}" in REAL:
            return importlib.import_module(f"{self.__name__}.{name}")
        if name in self._ov:
            return self._ov[name]
        v = dummy_cls(name)
        setattr(self, name, v)
        return v


class FakeEvent:
    _n = 0

    def __init__(self, label):
        self.label = label

    def record(self, stream=None):
        LOG.append(("record", self.label, stream))

    def wait(self, stream=None):
        LOG.append(("wait", self.label, stream))


class KernelStub:
    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__

    def __getitem__(self, grid):
        def launch(*args, **kw):
            LOG.append(("kernel", self.__name__, STREAM[0]))
            EMULATING[0] = True
            try:
                EMU[self.__name__](grid, *args, **kw)
            finally:
                EMULATING[0] = False
        return launch


def bf16_rne(x32):
    return x32.to(torch.bfloat16)          # torch CPU float->bf16 is round-to-nearest-even


def emu_gather(grid, n, inp, s0, s1, ids, i0, i1, w, w0, w1, idx, x0, x1, out, o0, o1, status, *rest,
               topk_num, BLOCK_D, num_warps, shared=None):
    ok = (int(status[0]) & 11) == 0
    d = out.shape[1]
    for t in range(n):
        acc = torch.zeros(d, dtype=torch.float32)
        if ok:
            for k in range(topk_num):
                if int(ids[t, k]) >= 0:
                    acc = torch.addcmul(acc, inp[int(idx[t, k])].float(), w[t, k].float())  # fp32 a + v*w
        routed = bf16_rne(acc)
        if shared is None:
            out[t] = routed
        else:
            out[t] = bf16_rne(routed.float() + shared[t].float())


def emu_gather_add(grid, n, inp, s0, s1, ids, i0, i1, w, w0, w1, idx, x0, x1, out, o0, o1, sh, h0, h1, status,
                   topk_num, BLOCK_D, num_warps):
    emu_gather(grid, n, inp, s0, s1, ids, i0, i1, w, w0, w1, idx, x0, x1, out, o0, o1, status,
               topk_num=topk_num, BLOCK_D=BLOCK_D, num_warps=num_warps, shared=sh)


EMU = {"_gated_gather": emu_gather, "_gated_gather_add": emu_gather_add}


def all_reduce(t):
    LOG.append(("AR", tuple(t.shape), STREAM[0]))
    AR_INPUTS.append(t.clone())
    return t


AR_INPUTS = []


class SharedExpertsOrderStub(IntEnum):
    NONE = 0
    NO_OVERLAP = 1
    MK_INTERNAL_OVERLAPPED = 2
    MULTI_STREAM_OVERLAPPED = 3


OVERRIDES = {
    "vllm.envs": {"VLLM_DISABLE_SHARED_EXPERTS_STREAM": False, "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD": 64},
    "vllm.platforms": {"current_platform": SimpleNamespace(is_cuda_alike=lambda: True, is_tpu=lambda: False,
                                                           is_cpu=lambda: False, is_rocm=lambda: False)},
    "vllm.utils.torch_utils": {"aux_stream": lambda: "aux", "current_stream": lambda: STREAM[0],
                               "_USE_LAYERNAME": False, "LayerName": dummy_cls("LayerName"),
                               "direct_register_custom_op": lambda **k: None},
    "vllm.v1.worker.ubatching": {"dbo_current_ubatch_id": lambda: 0},
    "vllm.model_executor.layers.fused_moe.runner.shared_fold_glue": {"maybe_fold_forward": lambda layer, x: None},
    "vllm.distributed": {"tensor_model_parallel_all_reduce": all_reduce},
    "vllm.forward_context": {"get_forward_context": lambda: SimpleNamespace(dp_metadata=None),
                             "is_forward_context_available": lambda: False},
    "vllm.logger": {"init_logger": lambda name: SimpleNamespace(debug_once=lambda *a, **k: None,
                                                                info_once=lambda *a, **k: None)},
    "vllm.model_executor.layers.fused_moe.runner.moe_runner_interface": {
        "MoERunnerInterface": type("MoERunnerInterface", (), {})},
    "vllm.triton_utils": {"triton": SimpleNamespace(jit=KernelStub, cdiv=lambda a, b: -(-a // b)),
                          "tl": Dummy()},
}

REAL = {}  # module name -> source path (exec'd for real)


class Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name == "vllm" or name.startswith("vllm."):
            return importlib.machinery.ModuleSpec(name, self, is_package=name not in REAL)
        return None

    def create_module(self, spec):
        if spec.name in REAL:
            return None
        return StubModule(spec.name, OVERRIDES.get(spec.name, {}))

    def exec_module(self, module):
        if module.__name__ in REAL:
            src = open(REAL[module.__name__]).read()
            exec(compile(src, REAL[module.__name__], "exec"), module.__dict__)


def fresh_import(paths):
    for k in list(sys.modules):
        if k == "vllm" or k.startswith("vllm."):
            del sys.modules[k]
    REAL.clear()
    REAL.update(paths)


@contextlib.contextmanager
def cuda_stream(s):
    old = STREAM[0]
    STREAM[0] = s
    try:
        yield
    finally:
        STREAM[0] = old


torch.cuda.stream = cuda_stream
torch.cuda.current_stream = lambda device=None: STREAM[0]


class OpLog(torch.overrides.TorchFunctionMode):
    SKIP = {"__get__", "size", "dim", "is_contiguous", "stride", "numel", "element_size", "data_ptr", "__len__",
            "__getitem__", "__setitem__", "clone", "float", "to", "zeros", "addcmul", "__index__", "__int__",
            "__bool__", "__format__", "item", "equal", "view", "__eq__", "__ne__", "__ge__", "__and__"}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, "__name__", str(func))
        if name not in self.SKIP and not EMULATING[0]:
            LOG.append(("torch", name, STREAM[0]))
        return func(*args, **(kwargs or {}))


EMULATING = [False]


# ------------------------------------------------------------------ build modules (base or patched)
def load(variant, flag):
    os.environ["DSV41_FC2_GATHER_ADD"] = "1" if flag else "0"
    root = os.path.join(KIT, "base")
    rend = os.path.join(KIT, "scratch", "render")
    runner_src = (os.path.join(root, "model_executor_layers_fused_moe_runner_moe_runner.py") if variant == "base"
                  else os.path.join(rend, "moe_runner.py"))
    gather_src = (os.path.join(root, "fc2_status_gated_gather_trial.py") if variant == "base"
                  else os.path.join(rend, "fc2_status_gated_gather_trial.py"))
    fresh_import({
        "vllm.model_executor.layers.fused_moe.runner.moe_runner": runner_src,
        "vllm.model_executor.layers.fused_moe.runner.shared_experts":
            os.path.join(root, "model_executor_layers_fused_moe_runner_shared_experts.py"),
        "vllm.model_executor.layers.fused_moe.runner.dsv41_moe_fold":
            os.path.join(KIT, "files/vllm/model_executor/layers/fused_moe/runner/dsv41_moe_fold.py"),
    })
    mr = importlib.import_module("vllm.model_executor.layers.fused_moe.runner.moe_runner")
    se = importlib.import_module("vllm.model_executor.layers.fused_moe.runner.shared_experts")
    # Athena gather: test-only relaxation of the CUDA device check (CPU tensors)
    gsrc = open(gather_src).read()
    assert gsrc.count("t.is_cuda and ") == 1
    gsrc = gsrc.replace("t.is_cuda and ", "")
    gmod = types.ModuleType("athena_fc2_gated_gather")
    exec(compile(gsrc, gather_src, "exec"), gmod.__dict__)
    return mr, se, gmod


# ------------------------------------------------------------------ fakes for the routed side
def make_runner(mr, se, gmod, *, athena=True, scale=1.0, reduced=False, hidden_pad=0, double_gather=False):
    R = mr.MoERunner.__new__(mr.MoERunner)
    hidden = 64
    cfg = SimpleNamespace(is_sequence_parallel=False, skip_final_all_reduce=False, tp_size=8, ep_size=1, dp_size=1,
                          pcp_size=1, hidden_dim=hidden, hidden_dim_unpadded=hidden, sp_size=1, use_ep=False,
                          moe_parallel_config=SimpleNamespace(enable_eplb=False, all2all_backend="x",
                                                              use_fi_nvl_two_sided_kernels=False,
                                                              use_all2all_kernels=False))
    status = torch.zeros(1, dtype=torch.int32)

    class Experts:
        def output_is_reduced(self):
            return reduced

    class QM:
        is_monolithic = False
        moe_kernel = Experts()
        supports_internal_mk = True
        skip_forward_padding = True
        has_unpadded_output = False
        mk_can_overlap_shared_experts = False
        topk_indices_dtype = None

    class Routed:
        quant_method = QM()

        def _ensure_moe_quant_config_init(self):
            pass

        def forward_modular(self, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
            m = x.shape[0]
            width = hidden + hidden_pad
            g = torch.Generator().manual_seed(1234 + m)
            m_sum = ((m * 6 + 31) // 32) * 32
            mm2 = (torch.randn(m_sum, width, generator=g) * 0.7).to(torch.bfloat16)
            inv = torch.randperm(m_sum, generator=g)[: m * 6].view(m, 6).to(torch.int32)
            output = torch.empty((m, width), dtype=torch.bfloat16)
            LOG.append(("kernel", "fc1+silu+fc2", STREAM[0]))
            if athena:
                gmod.gated_gather(mm2, topk_ids, topk_weights, inv, None, output, status, experimental=True)
                if double_gather:
                    gmod.gated_gather(mm2, topk_ids, topk_weights, inv, None, output, status, experimental=True)
            else:
                LOG.append(("kernel", "stock_gather", STREAM[0]))
                EMULATING[0] = True
                emu_gather(None, m, mm2, 0, 0, topk_ids, 0, 0, topk_weights, 0, 0, inv, 0, 0, output, 0, 0, status,
                           topk_num=6, BLOCK_D=0, num_warps=0)
                EMULATING[0] = False
            return output[:, :hidden] if hidden_pad else output

    class Router:
        def select_experts(self, hidden_states, router_logits, topk_indices_dtype, input_ids):
            m = hidden_states.shape[0]
            g = torch.Generator().manual_seed(99 + m)
            ids = torch.randint(0, 384, (m, 6), generator=g, dtype=torch.int32)
            w = torch.rand(m, 6, generator=g, dtype=torch.float32) * 0.5
            return w, ids

    class SharedMLP:
        def __call__(self, x):
            LOG.append(("kernel", "shared_mlp", STREAM[0]))
            g = torch.Generator().manual_seed(7 + x.shape[0])
            EMULATING[0] = True
            y = (torch.randn(x.shape[0], hidden, generator=g) * 0.4).to(torch.bfloat16)
            EMULATING[0] = False
            return y

    S = se.SharedExperts.__new__(se.SharedExperts)
    torch.nn.Module.__init__(S)
    S.enable_dbo = False
    S._output = [None, None]
    S._layer = SharedMLP()
    S._moe_config = cfg
    S._mk_can_overlap_shared_experts = lambda: False
    S._stream = "aux"
    S._input_ready_event = [FakeEvent("in0"), FakeEvent("in1")]
    S._output_ready_event = [FakeEvent("out0"), FakeEvent("out1")]

    R.moe_config = cfg
    R.router = Router()
    R.routed_input_transform = None
    R.routed_output_transform = None
    R.routed_scaling_factor = scale
    R.gate = None
    R.shared_expert_gate = None
    R.routed_experts = Routed()
    R.enable_dbo = False
    R._fse_fuse_gate = False
    R._shared_experts = S
    R.layer_name = "layer0"
    R._forward_entry = lambda h, r, s, i, name, unp: R._forward_impl(h, r, s, i)
    return R, S, status


def run(variant, flag, m, **kw):
    LOG.clear()
    AR_INPUTS.clear()
    mr, se, gmod = load(variant, flag)
    R, S, status = make_runner(mr, se, gmod, **{k: v for k, v in kw.items() if k != "fail"})
    if kw.get("fail"):
        status.fill_(1)
    x = (torch.randn(m, 64, generator=torch.Generator().manual_seed(5)) * 0.1).to(torch.bfloat16)
    with OpLog():
        out = R.forward(x, x)
    assert S._output[0] is None, "shared output not consumed exactly once"
    return list(LOG), AR_INPUTS[0].view(torch.int16).clone(), out


def expected_on_log(base_log):
    """base log with [gather kernel, torch add] -> [wait out0 on main, gather_add kernel]."""
    out = []
    i = 0
    while i < len(base_log):
        e = base_log[i]
        if e == ("kernel", "_gated_gather", "main"):
            out += [("wait", "out0", "main"), ("kernel", "_gated_gather_add", "main")]
        elif e[0] == "torch" and e[1] in ("add", "__add__"):
            pass
        else:
            out.append(e)
        i += 1
    return out


def main():
    rend = os.path.join(KIT, "scratch", "render")
    subprocess.run([sys.executable, os.path.join(KIT, "install.py"), "render", rend], check=True,
                   env=dict(os.environ, R2MB_VLLM=os.path.join(KIT, "scratch", "tree", "vllm"),
                            R2MB_FC2=os.path.join(KIT, "scratch", "tree", "fc2")), stdout=subprocess.DEVNULL)
    fails = 0

    def check(name, cond, detail=""):
        nonlocal fails
        print(f"{'PASS' if cond else 'FAIL'} {name} {detail}")
        fails += 0 if cond else 1

    scen = [
        ("overlap m=1", 1, {}), ("overlap m=6", 6, {}), ("overlap m=48", 48, {}),
        ("no-overlap m=96 (> stream threshold 64)", 96, {}),
        ("status failed m=4", 4, {"fail": True}),
    ]
    for name, m, kw in scen:
        b_log, b_ar, _ = run("base", False, m, **kw)
        o_log, o_ar, _ = run("patched", False, m, **kw)
        n_log, n_ar, _ = run("patched", True, m, **kw)
        check(f"[{name}] flag-off log == base", o_log == b_log)
        check(f"[{name}] flag-off AR input bits == base", torch.equal(o_ar, b_ar))
        check(f"[{name}] flag-on AR input bits == base", torch.equal(n_ar, b_ar))
        exp = expected_on_log(b_log)
        if m > 64:  # no-overlap: no event wait inserted
            exp = [e for e in exp if e != ("wait", "out0", "main")]
        check(f"[{name}] flag-on log == base with gather+add -> wait+gather_add", n_log == exp,
              "" if n_log == exp else f"\n  got {n_log}\n  exp {exp}")
        adds = [e for e in b_log if e[0] == "torch" and e[1] in ("add", "__add__")]
        check(f"[{name}] base has exactly one add", len(adds) == 1, str(adds))
    for name, m, kw in [("drafter/non-Athena experts", 6, {"athena": False}),
                        ("routed_scaling_factor 2.5", 6, {"scale": 2.5}),
                        ("fused output pre-reduced", 6, {"reduced": True}),
                        ("padded routed width (declined)", 6, {"hidden_pad": 32})]:
        b_log, b_ar, _ = run("base", False, m, **kw)
        n_log, n_ar, _ = run("patched", True, m, **kw)
        check(f"[{name}] flag-on == base (no fold)", n_log == b_log and torch.equal(n_ar, b_ar),
              "" if n_log == b_log else f"\n  got {n_log}\n  exp {b_log}")
    try:
        run("patched", True, 6, double_gather=True)
        check("[double gather] raises", False)
    except RuntimeError as e:
        check("[double gather] raises", "second Athena gather" in str(e), str(e))
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    sys.meta_path.insert(0, Finder())
    main()
