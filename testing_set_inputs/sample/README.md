# Sample inputs for dry runs

Minimal multimodal context files extracted from
[Explanation_Distillation/testing_set_inputs/DAAD_LLAMA](https://github.com/Sainithin-bit/Explanation_Distillation).

| File | Description |
|------|-------------|
| `optical_flow_output_sample.json` | Motion context (Mt) for one clip |
| `lane_change_sample_patch.json` | Spatial lane/road context (St) |
| `detections_sample.json` | CenterTrack-style detections |
| `videollama_sample.txt` | Video-level MLLM description (dv) |
| `test.csv` | Example split metadata |

## Full DRAMA / DAAD replication

For full benchmark reproduction:

1. Download the [DRAMA dataset](https://github.com/JeffLi0814/DRAMA).
2. Generate context JSON using the external tools documented in `context/README.md`
   (same pipeline as Explanation Distillation).
3. Point the scripts at your generated files instead of this sample folder.

## Default clip id for smoke test

```
337b694e-40f3-45f9-b0a7-88a1432ba0b3.mp4
```

Run:

```bash
bash scripts/run_zeroshot.sh
```
