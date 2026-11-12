"""r1-attn: mutation tests -- the plan checker in test_plan.py must reject broken variants of the helper
(missing join, missing fork wait, early join event, wrong host order, wrong WOA fork stream).

Run: python3 tests/test_mutations.py
"""
import os
import sys
import types
import unittest

KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(KIT, "tests"))
import test_plan as tp  # noqa: E402

SRC = open(os.path.join(KIT, "files/models/deepseek_v41/nvidia/dsv41_idxpar.py")).read()

MUTATIONS = {
    "no_med_join": ("            ev_med.wait(main)\n", "            pass\n"),
    "no_hi_join": ("        ev_hi.wait(main)\n", ""),
    "no_fork_wait_hi": ("            ev0.wait(hi)\n", ""),
    "no_fork_wait_low": ("            ev0.wait(low)\n", ""),
    "early_hi_record": (
        "            index_q, index_q_scale, index_weights_out = indexer.forward_q(\n"
        "                qr, qr_scale, indexer_weights, positions, a.indexer_rotary_emb\n"
        "            )\n"
        "            ev_hi.record(hi)\n",
        "            ev_hi.record(hi)\n"
        "            index_q, index_q_scale, index_weights_out = indexer.forward_q(\n"
        "                qr, qr_scale, indexer_weights, positions, a.indexer_rotary_emb\n"
        "            )\n"),
    "early_low_record": (
        "            q = a._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)\n"
        "            ev_low.record(low)\n",
        "            ev_low.record(low)\n"
        "            q = a._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)\n"),
    "dropped_compressed_insert": (
        "                compressor.insert_cache(latent, positions, a.rotary_emb)\n", ""),
}


def load_mutant(old, new):
    assert SRC.count(old) >= 1, old
    src = SRC.replace(old, new, 1)
    m = types.ModuleType("idxpar_mutant")
    m.__file__ = "mutant"
    exec(compile(src, "mutant", "exec"), m.__dict__)
    m._cuda = tp.fake_cuda
    m._get_forward_context = lambda: tp.FC
    m._breakable_active = lambda: tp._BCG.active
    m._warm_blas = lambda s, dev: tp.op("warm_blas", [], [f"warm@{s.name}"])
    return m


class TestMutants(unittest.TestCase):
    def _captured(self, helper):
        saved = tp.HELPER
        tp.HELPER = helper
        try:
            aux = [tp.FakeStream(name="aux0"), tp.FakeStream(name="aux1"), tp.FakeStream(name="aux2")]
            ls = [tp.make_layer(l, aux, True) for l in tp.SEQ]
            for a in ls:
                if a._dsv41_idxpar is not None:
                    a._dsv41_idxpar.ready = True
                    a._dsv41_idxpar.shared.warmed = True
            return ls, tp.run(tp.PATCHED_ON, ls)
        finally:
            tp.HELPER = saved

    def _stock(self):
        aux = [tp.FakeStream(name="aux0"), tp.FakeStream(name="aux1"), tp.FakeStream(name="aux2")]
        return tp.run(tp.STOCK, [tp.make_layer(l, aux, False) for l in tp.SEQ])

    def test_unmutated_passes(self):
        h = load_mutant("", "")
        _, log = self._captured(h)
        tp.check_hb(self, log)

    def test_each_mutant_is_caught(self):
        stock = self._stock()
        for name, (old, new) in MUTATIONS.items():
            with self.subTest(mutant=name):
                h = load_mutant(old, new)
                _, log = self._captured(h)
                caught = False
                try:
                    tp.check_hb(self, log)
                except AssertionError:
                    caught = True
                if not caught:
                    # dropping a kernel is caught by the kernel-multiset comparison
                    caught = tp.kernels(log) != tp.kernels(stock)
                self.assertTrue(caught, f"mutant {name} not detected")

    def test_wrong_host_order_moves_woa_fork(self):
        """If the indexer q chain were launched BEFORE the main q chain on the host, the main wq_b would be
        the last wq_b post-hook and the WOA fork would wait on the main wq_b (the skeptic's pitfall)."""
        src = SRC
        lo = src.index("        # 1) main q chain")
        mid = src.index("        # 2) kv-source layers")
        hi = src.index("        # 3) indexer q chain")
        end = src.index("        # Join every branch")
        swapped = src[:lo] + src[hi:end] + src[lo:mid] + src[mid:hi] + src[end:]
        h = load_mutant("", "")
        exec(compile(swapped, "mutant", "exec"), h.__dict__)
        h._cuda = tp.fake_cuda
        h._get_forward_context = lambda: tp.FC
        h._breakable_active = lambda: tp._BCG.active
        side2 = tp.FakeStream(name="l2pf_side2")
        last = {}

        def woa(module, stream):  # the real hook forks at the LAST wq_b of the block
            last["m"], last["s"] = module, stream

        tp.TR.hooks = [woa]
        try:
            _, log = self._captured(h)
        finally:
            tp.TR.hooks = []
        self.assertTrue(last["m"].endswith(".wq_b") and not last["m"].endswith(".indexer.wq_b"))
        self.assertEqual(last["s"].name, "aux1")
        # and the module-call order differs from stock (L2 recorder order changes)
        self.assertNotEqual(tp.calls(log, ".indexer"), tp.calls(self._stock(), ".indexer"))
        del side2


if __name__ == "__main__":
    unittest.main(verbosity=2)
