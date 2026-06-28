# Land Cover Classification

Semantic segmentation of satellite and aerial imagery using deep learning, trained on two public datasets: **DeepGlobe Land Cover** and **LandCover.ai**.

Built as an end-of-course group project (Phase 1 + Phase 2 combined deliverable).

---

## Results

| Dataset | Architecture | Encoder | Classes | Val mIoU |
|---|---|---|---|---|
| DeepGlobe | U-Net | ResNet-50 | 7 | **0.6371** |
| LandCover.ai | U-Net | ResNet-50 | 4 | **0.6814** |

### Per-Class IoU — DeepGlobe

| Class | IoU | Colour |
|---|---|---|
| Agriculture | 0.8085 | 🟡 Yellow |
| Forest | 0.7143 | 🟢 Green |
| Urban Land | 0.6826 | 🩵 Cyan |
| Water | 0.6624 | 🔵 Blue |
| Barren Land | 0.6060 | ⬜ White |
| Rangeland | 0.3487 | 🟣 Magenta |
| Unknown | 0.0000 | ⬛ Black (ignored) |

> **Key finding:** Rangeland (0.349 IoU) is the weakest class due to visual overlap with Agriculture and Barren Land in satellite imagery — a known challenge in DeepGlobe benchmarks.

---

## Repository Structure

```
land-cover-classification/
│
├── land_cover_classification.ipynb   # Main notebook (Phases 1 + 2)
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── .gitignore                        # Excludes weights, data, caches
│
└── results/
    ├── training_curves.png           # Loss + mIoU curves (30 epochs)
    ├── per_class_iou.png             # Per-class IoU bar chart
    ├── predictions.png               # Qualitative prediction grid
    ├── test_indistribution.png       # Gradio: DeepGlobe farmland image
    ├── test_distributionshift.png    # Gradio: Google Earth suburban image
    └── test_ood.png                  # Gradio: ground-level street photo
```

---

## 🛰️ Datasets

### DeepGlobe Land Cover Classification
- **Source:** [Kaggle — balraj98](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)
- **Size:** 803 training images at 2448×2448px
- **Classes:** Urban Land, Agriculture, Rangeland, Forest, Water, Barren Land, Unknown
- **Split used:** 85% train / 15% validation (stratified by random seed 42)

### LandCover.ai
- **Source:** [Kaggle — adrianboguszewski](https://www.kaggle.com/datasets/adrianboguszewski/landcoverai)
- **Paper:** Boguszewski et al., *LandCover.ai: Dataset for Automatic Mapping of Buildings, Woodlands, Water and Roads from Aerial Imagery*, CVPR Workshops 2021. [arXiv:2005.02264](https://arxiv.org/abs/2005.02264)
- **Size:** 41 large orthophoto tiles of rural Poland (25–50 cm/px resolution)
- **Classes:** Background, Building, Woodland, Water
- **Split used:** 85% train / 15% validation (stratified by random seed 42)

> **Note:** Dataset files are not included in this repository. Add both datasets via the Kaggle UI when running the notebook (see Reproduction steps below).

---

## Architecture

```
Input (512×512 RGB)
        │
   ResNet-50 Encoder          ← pretrained on ImageNet
   (frozen at 0.1× LR)
        │
   U-Net Decoder              ← skip connections at each scale
        │
   Segmentation Head          → N-class pixel predictions
```

- **Library:** [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- **Loss:** Combined CrossEntropy (0.5) + Dice (0.5) with inverse-frequency class weights
- **Optimizer:** AdamW with discriminative learning rates (encoder 10× lower than decoder)
- **Scheduler:** CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
- **Training:** 30 epochs, batch size 16 (8 per GPU), mixed precision (AMP), gradient accumulation ×2

---

## Hardware

- 2× NVIDIA Tesla T4 (15GB each) via Kaggle GPU accelerator
- CUDA 13.0 / PyTorch 2.x / Python 3.12

---

## Reproduction

### 1. Open the notebook on Kaggle

Upload `land_cover_classification.ipynb` to a new Kaggle notebook, or fork this repo and import it.

### 2. Add both datasets

In the Kaggle notebook UI, click **+ Add Data** and add:
- `balraj98/deepglobe-land-cover-classification-dataset`
- `adrianboguszewski/landcoverai`

### 3. Enable GPU

Settings → Accelerator → **GPU T4 x2**

### 4. Install dependencies

The first cell handles this automatically:
```bash
pip install segmentation-models-pytorch==0.3.3 albumentations==1.3.1 rasterio gradio torchmetrics
```

### 5. Run All

Run all cells top to bottom. Training takes approximately 2–3 hours on dual T4.

To **resume after a session disconnect** without retraining from scratch, see Section 2b in the notebook — the training loop automatically detects and resumes from `latest_model.pth` if it exists in `/kaggle/working`.

---

## Interactive Demo

Section 17 of the notebook launches a Gradio web interface:

```python
demo.launch(share=True)  # generates a public URL
```

**Features:**
- Upload any satellite/aerial image
- Predicted mask with class colour overlay
- Donut chart showing class breakdown with percentages
- Confidence heatmap (plasma colormap)
- OOD detection badge (✅ In-distribution / ⚠️ Uncertain / 🚫 Out-of-distribution)
- Class visibility filter to isolate individual classes

---

## Error Analysis Summary

Three-tier inference testing was performed to assess model behaviour across input types:

| Test | Image Type | Mean Confidence | Key Finding |
|---|---|---|---|
| In-distribution | DeepGlobe farmland | 0.745 | Correct, spatially coherent predictions. Agriculture/Water cleanest. |
| Distribution shift | Google Earth suburban (USA) | 0.631 | Confidence drops, Forest class collapses into Rangeland. Fragmented suburban tree cover not handled well. |
| Out-of-distribution | Ground-level street photo | 0.905 | **Overconfident wrong prediction** — 95.9% Urban Land. Classic softmax overconfidence on OOD input; known failure mode of segmentation models without explicit rejection class. |

> The OOD overconfidence result (high confidence on clearly wrong input) demonstrates a real limitation: softmax-based confidence is not a reliable OOD detector. Future work could address this with feature-space distance metrics or an explicit background/unknown rejection class.

---

## References

1. Demir, I. et al. *DeepGlobe 2018: A Challenge to Parse the Earth through Satellite Images.* CVPR Workshops, 2018.
2. Boguszewski, A. et al. *LandCover.ai: Dataset for Automatic Mapping of Buildings, Woodlands, Water and Roads from Aerial Imagery.* CVPR Workshops, 2021. [arXiv:2005.02264](https://arxiv.org/abs/2005.02264)
3. Ronneberger, O. et al. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI, 2015.
4. He, K. et al. *Deep Residual Learning for Image Recognition.* CVPR, 2016.
5. Yakubovskiy, P. *Segmentation Models PyTorch.* GitHub, 2020. https://github.com/qubvel/segmentation_models.pytorch

---

## Limitations

- Model trained exclusively on DeepGlobe's source regions (Thailand, Indonesia, India) — may not generalise well to very different climate zones or sensor types
- Rangeland class is consistently the weakest (IoU 0.349) due to visual overlap with Agriculture and Barren Land
- No explicit OOD rejection — model always outputs a class per pixel regardless of input validity
- LandCover.ai model trained on Polish aerial orthophotos only; different acquisition geometry from DeepGlobe satellite imagery

---

## 📄 License

Code in this repository is released under the MIT License.  
DeepGlobe dataset: see [original challenge terms](https://competitions.codalab.org/competitions/18468).  
LandCover.ai dataset: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
