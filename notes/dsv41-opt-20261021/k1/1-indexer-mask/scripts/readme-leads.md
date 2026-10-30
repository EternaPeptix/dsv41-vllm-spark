1. **Decode top-k above 8 rows on 1M-wide logits** (Phase D): persistent_topk costs 2.3 ms/step at 48 rows x 4K context and 6.0 ms at 32K, against 0.8 / 1.8 ms on 262K-wide logits. The live per_row kernel runs the same rows in 31 / 145 us per call. Candidate fixes:
   - (a) Pick per_row over persistent for long rows above 8 rows.
   - (b) Narrow the logits row stride for steps whose max context fits.
   Both are greedy-eq: per_row and persistent can choose different sets on an exact tie at the 512th score. CANON already fixes the order. exact.py is unaffected, because a single request (<= 6 rows) already takes per_row at 1M. This needs a conc A/B at 4K-32K contexts, which the bench prompts do not exercise.
2. **Fused CANON:** sort 512 int32 per row in one kernel instead of torch.sort + where + masked_fill + copy (28 us per call, x8 per step). The output bits are identical, so it is exact. Expected about -0.2 ms/step at 1u (+0.6%).
3. **Pinned AV profile for A/B work** (round 3 above): `DSV41_AV_COST_JSON=~/dsv41-stage/k1/1-indexer-mask/av-pin-i1C.json` holds the verify-cap calibration fixed across instances, which separates kernel time from AV re-calibration in matched C1.
