# Archived k1/3-rdma-loads bundle

This is Claude's original source/build/test bundle for the exact parallel-load kernel qualified in this deployment pass. All entries in `SHA256SUMS` were checked when exporting, including `build/rdma_ar_ext.so` (62daad6c...). No kernel source was changed or rebuilt in this pass.

The binary is specific to the tested Spark runtime and PyTorch/CUDA ABI. The original installation and multi-node test scripts depend on existing paths and tools on this fleet; this archive is not a generic installation recipe. GPU component tests require an idle test window and must not run alongside serving measurements.

The active deployment already has the binding and both binaries installed. Selection is through `DSV41_RDMA_PARLOAD`; see the parent report and launcher diff for the change and rollback. Numerical test results from Claude are preserved under the workspace's `work/claude-handoff-20261024/claude-evidence/component/` directory.
