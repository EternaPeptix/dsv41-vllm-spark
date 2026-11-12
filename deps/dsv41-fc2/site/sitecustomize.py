"""dsv41 A/B arm: Athena forty-layer packed FC2 trial WITHOUT SSD Engram (fail-closed)."""
import os, sys
if os.environ.get("ATHENA_FC2_SERVING_TRIAL") == "1":
    try:
        sys.path.insert(0, os.path.expanduser("~/dsv41-fc2"))
        if os.environ.get("DSV41_FC2_PDL", "0").strip() == "1":  # r2-moeback-g18
            import hashlib  # r2-moeback-g18
            _pdl_so = os.path.expanduser("~/dsv41-fc2/athena_tp8_fc2_packed_graph_pdl_sm121.so")  # r2-moeback-g18
            with open(_pdl_so, "rb") as _f: _pdl_sha = hashlib.sha256(_f.read()).hexdigest()  # r2-moeback-g18
            with open(_pdl_so + ".sha256") as _f: _pdl_want = _f.read().split()[0]  # r2-moeback-g18
            if _pdl_sha != _pdl_want: raise RuntimeError(f"DSV41_FC2_PDL: {_pdl_so} sha256 {_pdl_sha} != {_pdl_want}")  # r2-moeback-g18
            os.environ["ATHENA_FC2_GRAPH_SO"] = _pdl_so  # r2-moeback-g18
            os.environ.setdefault("DSV41_MOE_PDL_TRIGGER", "start")  # r2-moeback-g18
            print(f"DSV41_FC2_PDL on: {_pdl_so} {_pdl_sha[:12]} trigger={os.environ['DSV41_MOE_PDL_TRIGGER']}", file=sys.stderr, flush=True)  # r2-moeback-g18
        from athena_fc2_forty_layer_trial import install
        install()
    except BaseException:
        import traceback; traceback.print_exc(); os._exit(97)
