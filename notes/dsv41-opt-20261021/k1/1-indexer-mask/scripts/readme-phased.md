Not pursued. Rule: go ahead only if a <=262K graph set saves at least ~0.3 ms/step at rows 4-6 with length 300-4096. Measured: 25-131 us/step (4 rows: 131 / 25 us at len 300 / 4096; 6 rows: 84 / 96 us). The `max_seq_len=32768` bucket on 1M-wide logits is no better at 4-6 rows and slower at length >= 4096 (6 rows x 4096: 564 vs 482 us/step live), so it is dead too. Other findings:
  - At <= 8 rows on 1M-wide logits, live persistent_topk already takes the per_row path (20.4 us at 4 rows x 300 for both).
  - CANON alone (`_canon_topk_`: torch.sort + where + masked_fill + copy) costs a flat **~28 us per call** at every shape, x 8 index layers = **~0.22 ms/step**.
  - **Above 8 rows, persistent_topk's cost scales with the logits row width, not with the row lengths.** At 48 rows x len 4096 it takes 320.7 us at W=1M, 51.2 us at 131K, and 31.2 us for per_row. At 48 x 32768: 896 us vs per_row 145 us. Per-step model (5 ratio-1 layers at 1M + 3 ratio-2 at 512K, CANON included):

| rows | ctx (ratio-1 len) | live 1M us/step | <=262K graph set |
|---|---|---|---|
| 16 | 4096 | 963 | 438 |
| 48 | 300 | 337 | 273 |
| 48 | 4096 | 2345 | 797 |
| 48 | 32768 | 6048 | 1763 |
| 96 | 32768 | 11921 | 3291 |

  The bench conc runs use ~300-token contexts (48 rows x 300 = 0.34 ms/step), so this cost is invisible in the conc16/conc32 numbers here. Real multi-user traffic with 4K-32K contexts pays 1.5-4.3 ms/step at 48 rows (see Leads).
