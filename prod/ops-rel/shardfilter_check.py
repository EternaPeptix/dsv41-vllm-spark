"""rel-2 arm S check: which checkpoint shards hold an mtp.N.* tensor (header scan, as
filter_safetensors_files_by_weight_name does via safe_open().keys()), cross-checked with the index."""
import glob, json, os, re, struct, sys
d = sys.argv[1]
MTP = re.compile(r"mtp\.\d+\.")
unused = lambda n: MTP.match(n) is None
files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
kept, nmtp, nother_in_kept = [], 0, 0
for f in files:
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        keys = [k for k in json.loads(fh.read(n)) if k != "__metadata__"]
    if not all(unused(k) for k in keys):
        kept.append(os.path.basename(f))
        nmtp += sum(not unused(k) for k in keys)
        nother_in_kept += sum(unused(k) for k in keys)
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
idx_shards = sorted({v for k, v in idx.items() if MTP.match(k)})
idx_mtp = sum(1 for k in idx if MTP.match(k))
size = sum(os.path.getsize(os.path.join(d, k)) for k in kept)
print(json.dumps({"shards": len(files), "kept": kept, "kept_GB": round(size / 1e9, 2),
                  "mtp_tensors_in_kept": nmtp, "non_mtp_in_kept": nother_in_kept,
                  "index_mtp_shards": idx_shards, "index_mtp_tensors": idx_mtp,
                  "match_index": kept == idx_shards and nmtp == idx_mtp}))
