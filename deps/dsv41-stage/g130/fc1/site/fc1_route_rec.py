"""g130 fc1 route recorder (default off; DSV41_MOE_FC1_ROUTE_REC=1 on all ranks so they stay balanced, one diagnostic instance).

G1 routes have never been recorded: the 10-21 files are other prompts, and FC1 is inside CUDA graphs, so Python never sees a
replay's layout. A small Triton kernel runs just before each FC1 launch, is captured into the graphs with it, and appends one
fixed-size record to a GPU ring:
    [num_m_blocks, G, expert_0, live_rows_0, expert_1, live_rows_1, ...]   int32, one per m-block, REC words, zero-filled
Records are in call order, so 40 consecutive records from one step are that step's layers, in order.

Report: touch ~/dsv41-logs/fc1-route-rec.trigger on spark-one after the workload (mtime is the trigger; a later touch dumps again;
routes are the same on every rank, so one rank's dump is enough).
Writes ~/dsv41-logs/fc1-routes.<host>.<pid>.bin: 16-byte header <REC, CAP, total records, 0> (uint32) then the ring, in call
order when not wrapped. tools/routes_to_hist.py reads it.

Cost: one kernel per FC1 launch (a few us) and 64 MiB of GPU memory. It reads only the layout and writes only its ring, so served
bytes cannot change; the extra time does change step timing, so an instance with the recorder on gives no G1 number.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import threading
import time

import torch
import triton
import triton.language as tl

REC = 128
CAP = 128 * 1024  # 64 MiB of int32: 3,276 steps at 40 layers


@triton.jit
def _rec_kernel(layout, nmb, ring, cursor, REC_W: tl.constexpr, CAP_N: tl.constexpr):
    n = tl.atomic_add(cursor, 1)
    base = ring + (n % CAP_N) * REC_W
    tl.store(base + 0, nmb)
    g = 0
    limit = tl.minimum(nmb, (REC_W - 2) // 2)
    for m in range(0, limit):
        r = tl.arange(0, 32)
        x = tl.load(layout + m * 32 + r)
        e = tl.load(layout + m * 32)
        c = tl.sum(tl.where(x >= 0, 1, 0), axis=0)
        tl.store(base + 2 + 2 * m, e)
        tl.store(base + 3 + 2 * m, c)
        g += tl.where(e >= 0, 1, 0)
    tl.store(base + 1, g)


class RouteRec:
    def __init__(self):
        assert not torch.cuda.is_current_stream_capturing()
        self.device = torch.cuda.current_device()
        self.ring = torch.zeros((CAP * REC,), dtype=torch.int32, device="cuda")
        self.cursor = torch.zeros((1,), dtype=torch.int32, device="cuda")
        self._side = torch.cuda.Stream()
        # compile now, eagerly, outside any capture
        layout = torch.full((32,), -1, dtype=torch.int32, device="cuda")
        layout[:3] = 0
        self._launch(layout)
        torch.cuda.synchronize()
        self.ring.fill_(0)
        self.cursor.fill_(0)
        self._start_reporter()

    def _launch(self, layout):
        _rec_kernel[(1,)](layout, layout.shape[0] // 32, self.ring, self.cursor, REC_W=REC, CAP_N=CAP, num_warps=1)

    def observe(self, args) -> None:
        """Called from TargetM32Kernel.gemm. FC1 target geometry only: anything else returns untouched."""
        try:
            (_aq, _sfa), (bw, _sfb), d, layout = args
        except (TypeError, ValueError):
            return
        if int(d.shape[1]) != 576 or int(bw.shape[0]) != 384 or layout.shape[0] % 32 != 0:
            return
        self._launch(layout)

    def _start_reporter(self) -> None:
        root = Path(os.environ.get("DSV41_MOE_FC2_D05_VERIFY_DIR", str(Path.home() / "dsv41-logs")))
        trigger = root / "fc1-route-rec.trigger"
        out = root / f"fc1-routes.{socket.gethostname()}.{os.getpid()}.bin"

        def loop():
            seen = None
            try:
                seen = trigger.stat().st_mtime
            except OSError:
                pass
            while True:
                time.sleep(2.0)
                try:
                    mtime = trigger.stat().st_mtime
                except OSError:
                    continue
                if mtime == seen:
                    continue
                seen = mtime
                try:
                    with torch.cuda.device(self.device), torch.cuda.stream(self._side):
                        ring = self.ring.cpu()
                        n = int(self.cursor.cpu()[0])
                    head = torch.tensor([REC, CAP, n, 0], dtype=torch.int32)
                    tmp = out.with_suffix(".tmp")
                    with open(tmp, "wb") as f:
                        f.write(head.numpy().tobytes())
                        f.write(ring.numpy().tobytes())
                    tmp.replace(out)
                    out.with_suffix(".json").write_text(json.dumps(dict(records=n, time=time.time(), REC=REC, CAP=CAP)))
                except Exception as exc:  # diagnostics must never kill serving
                    try:
                        out.with_suffix(".err").write_text(repr(exc))
                    except OSError:
                        pass

        threading.Thread(target=loop, name="fc1-route-rec", daemon=True).start()
