# DriveSafe

Official repository for **DriveSafe: A Framework for Risk Detection and Safety Suggestions in Driving Scenarios** (ICRA 2026).

**Authors:** Sainithin Artham · Avijit Dasgupta · Shankar Gangisetty · C. V. Jawahar

**Model checkpoints (Hugging Face):** [SAINITHIN/DriveSafe_LLaMA_Adapter](https://huggingface.co/SAINITHIN/DriveSafe_LLaMA_Adapter) — LLaMA-Adapter weights for DriveSafe-Finetuned (`DriveSafe_LLaMA_Adapter_8B.pth`).

---

## Overview

DriveSafe is a two-stage framework for driving risk assessment:

1. **Caption generation** — fuse spatial (St), motion (Mt), depth (Dt), and video description (dv) cues into a geometry-aware scene caption Cv.
2. **Risk assessment & safety suggestion** — an LLM predicts risk label, risk caption, keywords, and bounding box; keywords map to actionable safety suggestions via Table I in the paper.

```
Video → [St, Mt, Dt, dv] → LLM → Caption Cv → LLM → (risk, bbox, keywords) → Safety suggestion
```

This repo reuses the multimodal context pipeline from our prior work
[Explanation Distillation (WACV 2026)](https://github.com/Sainithin-bit/Explanation_Distillation)
and adds DriveSafe-specific prompts, risk parsing, safety mapping, and evaluation.

**Paper (local):** place `ICRA26_DriveSafe_CRC.pdf` in `docs/` or use your copy from Downloads.

---

## Repository layout

| Path | Role |
|------|------|
| `context/` | Optical flow, JSON loaders, depth helpers |
| `caption_generation/` | Stage 1 — multimodal caption generation |
| `risk_assessment/` | Stage 2 — risk inference + safety mapping |
| `evaluation/` | Caption, grounding, and safety metrics |
| `finetuning/` | LLaMA-Adapter triplet prep + training config |
| `configs/` | Prompt templates + safety keyword map (Table I) |
| `scripts/run_zeroshot.py` | End-to-end zero-shot demo |
| `testing_set_inputs/sample/` | Minimal JSON sample from Explanation Distillation |

---

## Installation

```bash
git clone https://github.com/Sainithin-bit/DriveSafe.git
cd DriveSafe
conda env create -f environment.yml
conda activate drivesafe
```

**GPU:** CUDA strongly recommended for `meta-llama/Meta-Llama-3.1-8B-Instruct`.

**Hugging Face:** Accept the Meta Llama 3.1 license and login:

```bash
pip install huggingface_hub
huggingface-cli login
```

### External context tools (shared with Explanation Distillation)

| Cue | Tool | ED reference |
|-----|------|--------------|
| St (spatial) | HybridNets | `Explanation_Distillation/RLM/` |
| Mt (motion) | OpenCV Farnebäck | `Explanation_Distillation/OFM.py` → `context/optical_flow.py` |
| Detections | CenterTrack | `Explanation_Distillation/Surrounding_context/` |
| dv (video desc) | VideoLLaMA | `Explanation_Distillation/Description_generation/` |
| Dt (depth) | DepthAnything-v2 | See paper Sec. IV-B |

See `context/README.md` for details.

---

## Quick start (sample dry run)

Uses minimal sample JSON under `testing_set_inputs/sample/`:

```bash
bash scripts/run_zeroshot.sh
```

Or step by step:

```bash
# Stage 1 — caption generation
python caption_generation/generate_captions.py \
  --clip-id 337b694e-40f3-45f9-b0a7-88a1432ba0b3.mp4 \
  --spatial-json testing_set_inputs/sample/lane_change_sample_patch.json \
  --motion-json testing_set_inputs/sample/optical_flow_output_sample.json \
  --detections-json testing_set_inputs/sample/detections_sample.json \
  --videollama-txt testing_set_inputs/sample/videollama_sample.txt \
  --output outputs/captions

# Stage 2 — risk + safety suggestion
python risk_assessment/infer_risk.py \
  --caption-file outputs/captions/337b694e-40f3-45f9-b0a7-88a1432ba0b3.json \
  --output outputs/risk
```

Outputs:

- `outputs/captions/<clip>.json` — generated caption + context bundle
- `outputs/risk/<clip>_risk.json` — risk label, caption, keywords, bbox, safety suggestion

---

## Full DRAMA reproduction

1. Download [DRAMA](https://github.com/JeffLi0814/DRAMA).
2. Generate context JSON with HybridNets, CenterTrack, optical flow, DepthAnything-v2, and VideoLLaMA (same as Explanation Distillation).
3. Run Stage 1 and Stage 2 over the split CSV.
4. Evaluate with `evaluation/run_eval.py`.

### Expected prediction JSON shape

```json
{
  "clip_id": "example.mp4",
  "caption": "...",
  "prediction": {
    "risk": "Yes",
    "risk_caption": "...",
    "keywords": ["Stopped vehicle"],
    "bounding_box": [100, 200, 300, 400],
    "safety_suggestion": "(Must) Stop"
  }
}
```

---

## Safety keyword mapping

Keyword → suggestion mapping is in `configs/safety_keyword_map.json` (paper Table I).
Implemented in `risk_assessment/safety_mapping.py`.

---

## Finetuning (DriveSafe-Finetuned)

Paper settings (Sec. IV-B):

- Framework: LLaMA-Adapter
- Base model: Meta-Llama-3.1-8B-Instruct
- Batch size 4, lr 2e-5, weight decay 0.01, 5 epochs
- Mixed precision + gradient checkpointing

```bash
python finetuning/prepare_triplets.py \
  --annotations /path/to/drama_annotations.json \
  --output outputs/finetuning/triplets.jsonl

python finetuning/train_llama_adapter.py \
  --triplets outputs/finetuning/triplets.jsonl
```

Integrate the generated config with [OpenGVLab/LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter).

### Download finetuned adapter

```bash
huggingface-cli download SAINITHIN/DriveSafe_LLaMA_Adapter \
  DriveSafe_LLaMA_Adapter_8B.pth adapter_config.json training_config.json \
  --local-dir checkpoints/DriveSafe_LLaMA_Adapter
```

Upload or refresh weights on the Hub:

```bash
python scripts/upload_adapter_to_hf.py \
  --checkpoint /path/to/DriveSafe_LLaMA_Adapter_8B.pth
```

---

## Evaluation

```bash
python evaluation/run_eval.py \
  --predictions outputs/predictions.json \
  --ground-truth outputs/ground_truth.json
```

Metrics:

- Caption: BLEU-1, ROUGE-L (extend with pycocoevalcap for full Table II metrics)
- Grounding: Mean IoU, Acc@0.5
- Safety: accuracy, weighted F1 (8 classes, excluding NA)

---

## Extract motion context from a video

```bash
python context/optical_flow.py \
  --video /path/to/clip.mp4 \
  --output outputs/context/motion.json
```

---
## Citation

```bibtex
@inproceedings{artham2026drivesafe,
  title     = {{DriveSafe}: A Framework for Risk Detection and Safety Suggestions in Driving Scenarios},
  author    = {Artham, Sainithin and Dasgupta, Avijit and Gangisetty, Shankar and Jawahar, C. V.},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026}
}
```

---

## Contact

**Sainithin Artham** · [sainithin.artham@gmail.com](mailto:sainithin.artham@gmail.com)

---

## License

Specify in `LICENSE` before public release.
