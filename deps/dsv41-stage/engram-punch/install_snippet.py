import hashlib, os, shutil, sys
L = os.path.expanduser("~/dsv41-tp8.sh"); SNIP = os.path.expanduser("~/dsv41-stage/engram-punch/tp8-startup-check.snippet.sh")
src = open(L).read()
if hashlib.sha256(src.encode()).hexdigest()[:8] != "85189840":
    sys.exit("launcher sha is not 85189840; refusing")
anchor = 'source "$HOME/venvs/vllm032/bin/activate"\n'
assert src.count(anchor) == 1 and "Engram row check" not in src
shutil.copy2(L, L + ".bak-punch")
snip = open(SNIP).read()
if not snip.endswith("\n"): snip += "\n"
out = src.replace(anchor, anchor + snip)
open(L, "w").write(out)
print(os.uname().nodename, hashlib.sha256(out.encode()).hexdigest()[:8], hashlib.sha256(open(L + ".bak-punch").read().encode()).hexdigest()[:8])
