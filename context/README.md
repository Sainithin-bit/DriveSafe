# Context extraction for DriveSafe

DriveSafe caption generation uses four contextual signals (paper Eq. 1):

| Signal | Symbol | Source | Explanation Distillation reference |
|--------|--------|--------|----------------------------------|
| Spatial (road/lane) | St | HybridNets | `RLM/README.md` |
| Motion (optical flow) | Mt | OpenCV Farnebäck | `OFM.py` → `context/optical_flow.py` |
| Depth (per object) | Dt | DepthAnything-v2 | new in this repo (`context/depth.py`) |
| Video description | dv | VideoLLaMA / MLLM | `Description_generation/` |

## Included in this repo

- `optical_flow.py` — extract motion JSON from videos
- `loaders.py` — load JSON context files for LLM prompts
- `depth.py` — depth JSON loader / stub

## External setup (same as Explanation Distillation)

### Road and lane segmentation (St)

Follow [Explanation_Distillation/RLM/README.md](https://github.com/Sainithin-bit/Explanation_Distillation) (HybridNets).

Output: `lane_change_*_patch.json` keyed by video id.

### Object detections (for depth + bbox grounding)

Follow [Explanation_Distillation/Surrounding_context/README.md](https://github.com/Sainithin-bit/Explanation_Distillation) (CenterTrack).

Output: `detections_*.json`.

### Video description (dv)

Follow [Explanation_Distillation/Description_generation/README.md](https://github.com/Sainithin-bit/Explanation_Distillation) (VideoLLaMA).

Output: `videollama_*.txt` or JSON captions.

### Depth (Dt)

Install [DepthAnything-V2](https://github.com/DepthAnything/Depth-Anything-V2) and export per-object depth aligned with detection boxes.

## Quick motion extraction

```bash
python context/optical_flow.py --video /path/to/clip.mp4 --output outputs/context/motion.json
```

## Sample inputs

See `testing_set_inputs/sample/` for minimal JSON structure copied from Explanation_Distillation DAAD examples.
