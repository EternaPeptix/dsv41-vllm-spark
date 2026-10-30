#!/usr/bin/env python3
"""balloon.py: keep kcompactd's proactive compaction from firing without root.

Proactive compaction runs when the node's fragmentation score (share of free memory that is NOT in
>=2 MB blocks) exceeds 100 - vm.compaction_proactiveness + 10 (= 90 by default). On a Spark serving
a model, most memory is pinned GPU memory that compaction can never move, so every pass fails, and
while it runs it stalls that rank's GPU kernels and with them the whole TP group.

This process allocates 4 KB pages (MADV_NOHUGEPAGE) in chunks. The buddy allocator serves those from
the smallest free blocks first, so the fragments get absorbed and what remains free is mostly large
blocks, i.e. a low score. Safety: it never lets free memory drop below --reserve-gb, gives chunks
back when free memory falls below that, and stops growing if a chunk consumed large blocks. Run it
only after the server has finished loading; kill it (SIGTERM) to return all memory.
"""
import argparse, ctypes, mmap, os, signal, sys, time

libc = ctypes.CDLL("libc.so.6", use_errno=True)
MADV_NOHUGEPAGE = 15
PAGE = 4096


def buddy():
    """(free bytes, bytes in >= order-9 blocks) for the Normal zone(s)."""
    free = big = 0
    for line in open("/proc/buddyinfo"):
        parts = line.split()
        if "Normal" not in parts:
            continue
        counts = [int(x) for x in parts[parts.index("Normal") + 1:]]
        for order, c in enumerate(counts):
            pages = c << order
            free += pages
            if order >= 9:
                big += pages
    return free * PAGE, big * PAGE


def score():
    free, big = buddy()
    return (100.0 * (free - big) / free if free else 0.0), free, big


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=70.0, help="aim for a fragmentation score below this")
    ap.add_argument("--reserve-gb", type=float, default=2.5, help="never leave less free memory than this")
    ap.add_argument("--chunk-mb", type=int, default=256)
    ap.add_argument("--max-gb", type=float, default=16.0)
    a = ap.parse_args()
    chunk = a.chunk_mb << 20
    reserve = int(a.reserve_gb * (1 << 30))
    chunks = []
    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("v", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("v", True))
    grow = True
    last_log = 0
    while not stop["v"]:
        s, free, big = score()
        held = len(chunks) * chunk
        if free < reserve and chunks:  # give memory back first
            chunks.pop().close()
            print(f"release: free={free/2**30:.1f}G held={held/2**30:.1f}G score={s:.0f}", flush=True)
            continue
        if grow and s > a.target and free - chunk > reserve and held + chunk <= a.max_gb * (1 << 30):
            m = mmap.mmap(-1, chunk, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            buf = ctypes.c_char.from_buffer(m)
            libc.madvise(ctypes.c_void_p(ctypes.addressof(buf)), ctypes.c_size_t(chunk), MADV_NOHUGEPAGE)
            del buf  # no exported pointer may outlive this, or close() fails
            for off in range(0, chunk, PAGE):  # fault every page in
                m[off] = 1
            s2, free2, big2 = score()
            chunks.append(m)
            if big - big2 > chunk // 2:  # we are splitting large blocks now: undo and stop growing
                chunks.pop().close()
                grow = False
                print(f"stop growing: chunk ate large blocks (big {big/2**30:.2f}->{big2/2**30:.2f}G)", flush=True)
            continue
        now = time.time()
        if now - last_log > 60:
            print(f"hold: held={held/2**30:.1f}G free={free/2**30:.1f}G big={big/2**30:.1f}G score={s:.0f}", flush=True)
            last_log = now
        if s > a.target + 15 and not grow and free - chunk > reserve:
            grow = True  # fragments came back (e.g. page cache churn): try again
        time.sleep(2)
    for m in chunks:
        m.close()
    print("released all", flush=True)


if __name__ == "__main__":
    main()
