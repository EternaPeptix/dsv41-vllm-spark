"""dsv41 A/B arm: Athena forty-layer packed FC2 trial WITHOUT SSD Engram (fail-closed)."""
import os, sys
if os.environ.get("ATHENA_FC2_SERVING_TRIAL") == "1":
    try:
        sys.path.insert(0, os.path.expanduser("~/dsv41-fc2"))
        from athena_fc2_forty_layer_trial import install
        install()
    except BaseException:
        import traceback; traceback.print_exc(); os._exit(97)
