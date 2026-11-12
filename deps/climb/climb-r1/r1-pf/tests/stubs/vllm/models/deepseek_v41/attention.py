"""Test stub: the live DeepseekV4Attention._run_parallel_input_projections runs the compressor torch.mm, the
indexer weights_proj and the qkv_a shard GEMM + all-gather (r3). The harness installs SIM, which emits the
same event protocol the live recorders produce."""
SIM = None


class DeepseekV4Attention:
    def __init__(self, layer, compressor=None):
        self.layer = layer
        self.compressor = compressor
        self.prefix = f"model.layers.{layer}.attn"

    def _run_parallel_input_projections(self, hidden_states):
        return SIM.input_projections(self, hidden_states)
