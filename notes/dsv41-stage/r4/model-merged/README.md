# r4 model-merged (conflict resolution)
small-kernels (patched sha b403734f) and prefill-sp (patched sha 67ea8ef5) both patch
vllm/models/deepseek_v41/nvidia/model.py from the same live original 5fe8b4a7.
`git merge-file` gave exactly one conflict: both insert a flag block right after `logger = init_logger(__name__)`.
Resolved by keeping both blocks (small-kernels first, then prefill-sp). Checked on the Mac:
every added line of both diffs is present in the merge (19 + 247, 0 missing); ast.parse OK.
Merged sha256 is in SHA256SUMS. dsv41_r4_small.py is the unchanged small-kernels module (02f94f29).

Install (services stopped):   bash install.sh     (common/r4install.sh: preflight 8/8 orig, .bak-r4-model-merged, verify 8/8)
Uninstall:                    bash uninstall.sh
Do NOT also run small-kernels/install.sh or prefill-sp/install.sh (both would abort on the sha anyway).
Before serving, on one Spark CPU: ~/venvs/vllm032/bin/python -m py_compile <site>/models/deepseek_v41/nvidia/model.py
