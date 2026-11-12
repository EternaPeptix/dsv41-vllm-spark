"""Test stub: DSparkSpeculator with the live _generate_draft / _sample_sequential signatures; SIM emits the
drafter's event protocol (collectives, weight reads, wq_b boundaries, wo_a reads)."""
SIM = None


class DSparkSpeculator:
    def __init__(self, model, greedy_state=None):
        self.model = model
        self._sharded_markov = greedy_state

    def _generate_draft(self, num_reqs, num_tokens_padded, attn_metadata, slot_mappings, num_tokens_across_dp,
                        cudagraph_runtime_mode=None):
        SIM.drafter_backbone(self, num_tokens_padded)
        self._sample_sequential(num_reqs, None)

    def _sample_sequential(self, num_reqs, head_hidden):
        SIM.drafter_markov(self, num_reqs)
