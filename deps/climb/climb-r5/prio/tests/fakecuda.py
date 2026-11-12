"""CPU stand-in for torch.cuda streams / events with a vector-clock race checker (r5-prio tests).

Every "kernel" is an op on the current stream that reads and writes named fake tensors. A read is legal only
if the reading stream's vector clock already covers the op that last wrote the tensor (stream order, event
record/wait and wait_stream are the only ways clocks advance), i.e. exactly the happens-before relation CUDA
guarantees, eagerly or in a captured graph. Write-after-read is checked the same way (a write must happen after
every earlier read of the tensor on other streams). join_check() asserts that the main stream's clock covers
every op of every stream (no open branch at the end of a forward / capture).
"""
from __future__ import annotations

import contextlib
import itertools


class Race(AssertionError):
    pass


class World:
    def __init__(self):
        self.ids = itertools.count()
        self.log = []          # (kind, name, stream_id, priority, extra)
        self.races = []
        self.capturing = False
        self.streams = []
        self.default = Stream(self, priority=0, name="main")
        self.stack = [self.default]

    # -- torch.cuda API surface used by the kit ---------------------------------------------------------
    def current_stream(self, device=None):
        return self.stack[-1]

    @contextlib.contextmanager
    def stream(self, s):
        self.stack.append(s)
        try:
            yield s
        finally:
            self.stack.pop()

    def kernel(self, name, reads=(), writes=()):
        s = self.current_stream()
        for t in reads:
            w = t.writer
            if w is not None and s.vc.get(w[0], 0) < w[1]:
                self.races.append(f"RAW: {name} on {s.name} reads {t.name} before writer {w}")
        s.tick()
        stamp = (s.id, s.vc[s.id])
        for t in writes:
            for r in t.readers:
                if r[0] != s.id and s.vc.get(r[0], 0) < r[1]:
                    self.races.append(f"WAR: {name} on {s.name} overwrites {t.name} before reader {r}")
            if t.writer is not None and t.writer[0] != s.id and s.vc.get(t.writer[0], 0) < t.writer[1]:
                self.races.append(f"WAW: {name} on {s.name} overwrites {t.name} before writer {t.writer}")
            t.writer = stamp
            t.readers = []
        for t in reads:
            t.readers.append(stamp)
        self.log.append(("k", name, s.id, s.priority, (stamp, dict(s.vc))))
        return stamp

    def kentry(self, name, nth=0):
        """(stream_id, priority, stamp, vc) of the nth kernel called name."""
        hits = [e for e in self.log if e[0] == "k" and e[1] == name]
        e = hits[nth]
        return e[2], e[3], e[4][0], e[4][1]

    def before(self, a, b, na=0, nb=0):
        """True iff kernel a happens-before kernel b (b's stream clock covered a when b ran)."""
        _sa, _pa, sta, _ = self.kentry(a, na)
        _sb, _pb, _stb, vcb = self.kentry(b, nb)
        return vcb.get(sta[0], 0) >= sta[1]

    def join_check(self, main=None):
        main = main or self.default
        for s in self.streams:
            if s is main:
                continue
            n = s.vc.get(s.id, 0)
            if main.vc.get(s.id, 0) < n:
                raise Race(f"stream {s.name} has {n - main.vc.get(s.id, 0)} op(s) not joined into {main.name}")

    def assert_no_races(self):
        if self.races:
            raise Race("; ".join(self.races[:6]))

    def kernels(self):
        return [(e[1], e[2]) for e in self.log if e[0] == "k"]


class Stream:
    _least, _greatest = 0, -5

    def __init__(self, world=None, device=None, priority=0, name=None):
        self.w = world if world is not None else WORLD[0]
        self.id = next(self.w.ids)
        lo, hi = Stream._least, Stream._greatest
        self.priority = max(hi, min(lo, int(priority)))  # torch clamps into the device range
        self.name = name or f"s{self.id}"
        self.vc = {}
        self.cuda_stream = 0x1000 + self.id
        self.w.streams.append(self)

    @staticmethod
    def priority_range():
        return Stream._least, Stream._greatest

    def tick(self):
        self.vc[self.id] = self.vc.get(self.id, 0) + 1

    def _merge(self, vc):
        for k, v in vc.items():
            if self.vc.get(k, 0) < v:
                self.vc[k] = v

    def wait_event(self, ev):
        self.w.log.append(("wait_event", ev.name, self.id, self.priority, dict(ev.vc)))
        self._merge(ev.vc)

    def wait_stream(self, other):
        self.w.log.append(("wait_stream", other.name, self.id, self.priority, None))
        self._merge(other.vc)

    def __repr__(self):
        return f"<Stream {self.name} prio {self.priority}>"


class Event:
    _n = itertools.count()

    def __init__(self, *a, **k):
        self.w = WORLD[0]
        self.vc = {}
        self.name = f"ev{next(Event._n)}"
        self.recorded = False

    def record(self, stream=None):
        s = stream if stream is not None else self.w.current_stream()
        self.vc = dict(s.vc)
        self.recorded = True
        self.w.log.append(("record", self.name, s.id, s.priority, dict(s.vc)))

    def wait(self, stream=None):
        s = stream if stream is not None else self.w.current_stream()
        s.wait_event(self)


class T:
    """Fake tensor: identity for the race checker; shape for the code under test."""
    _n = itertools.count()

    def __init__(self, name=None, shape=(4, 8), device=0, dtype="bf16"):
        self.name = name or f"t{next(T._n)}"
        self.shape = tuple(shape)
        self.device = device
        self.dtype = dtype
        self.writer = None
        self.readers = []

    def flatten(self, *a):
        return self  # same storage: keeps the writer/reader stamps

    def view(self, *a):
        return self

    def __getitem__(self, idx):
        return self

    def record_stream(self, s):
        WORLD[0].log.append(("record_stream", self.name, s.id, s.priority, None))

    @property
    def ndim(self):
        return len(self.shape)

    def stride(self, i=None):
        return 1

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n


WORLD = [None]


def install(monkeypatch_torch):
    """Replace the torch.cuda attributes the kit touches; returns the World."""
    import torch

    w = World()
    WORLD[0] = w
    saved = {k: getattr(torch.cuda, k, None) for k in
             ("Stream", "Event", "stream", "current_stream", "current_device", "is_current_stream_capturing")}
    torch.cuda.Stream = Stream
    torch.cuda.Event = Event
    torch.cuda.stream = w.stream
    torch.cuda.current_stream = w.current_stream
    torch.cuda.current_device = lambda: 0
    torch.cuda.is_current_stream_capturing = lambda: w.capturing
    monkeypatch_torch.append(saved)
    return w


def uninstall(saved_list):
    import torch

    for saved in saved_list:
        for k, v in saved.items():
            if v is None:
                if hasattr(torch.cuda, k):
                    delattr(torch.cuda, k)
            else:
                setattr(torch.cuda, k, v)
    saved_list.clear()
