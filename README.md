# Land Cover Classification

Semantic segmentation of satellite and aerial imagery using deep learning, trained on two public datasets: **DeepGlobe Land Cover** and **LandCover.ai**.

---

## Visual Results

### Training & Benchmarking
![Training Curves](results/training_curves.PNG)
*Figure 1: Training progress showing steady convergence of Loss and mIoU.*

![Per-Class IoU](results/per_class_IoU.PNG)
*Figure 2: Per-class validation IoU breakdown for the DeepGlobe dataset.*

### Qualitative Predictions
![Predictions](results/prediction1.PNG)
*Figure 3: Sample validation grid showcasing model segmentation performance.*

### Interactive Demo (Gradio)
![Gradio App](results/app_screenshot.PNG)
*Figure 4: The Gradio web interface, featuring dynamic visualizations of model predictions, confidence maps, and class distribution charts.*

---

## Performance Summary

| Dataset | Architecture | Encoder | Classes | Best Val mIoU | Best Epoch |
|---|---|---|---|---|---|
| DeepGlobe | U-Net | ResNet-50 | 7 | **0.6370** | 29 / 30 |
| LandCover.ai | U-Net | ResNet-50 | 4 | **0.7982** | 27 / 30 |

### Per-Class IoU — DeepGlobe Validation Set

| Class | IoU | Mask Colour |
|---|---|---|
| Agriculture | 0.8085 | Yellow |
| Forest | 0.7148 | Green |
| Urban Land | 0.6818 | Cyan |
| Water | 0.6623 | Blue |
| Barren Land | 0.6065 | White |
| Rangeland | 0.3482 | Magenta |
| Unknown | 0.0000 | Black (ignored in training) |

**Key finding:** Rangeland (IoU 0.348) is the weakest class due to visual ambiguity with Agriculture and Barren Land in satellite imagery. This is corroborated by qualitative inference where mixed scrubland regions are consistently the least spatially coherent in the predicted mask.

---

## Repository Structure

```text
land-cover-classification/
│
├── land_cover_classification.ipynb   # Main notebook (Phase 1 + Phase 2)
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── .gitignore                        # Excludes weights, datasets, caches
│
└── results/                          # Visualization & Inference artifacts
    ├── training_curves.PNG           # Loss and mIoU curves
    ├── per_class_IoU.PNG             # Class-wise performance metrics
    ├── prediction1.PNG               # Sample validation grid
    ├── app_screenshot.PNG            # Gradio demo interface
    │
    ├── 195553_sat.jpg                # In-distribution: DeepGlobe sample
    ├── 195553_predicted_mask.webp
    ├── 195553_confidence_map.webp
    ├── 195553_donut_chart.webp
    │
    ├── iowa_sat_image.PNG            # Distribution shift: Iowa sample
    ├── iowa_predicted_mask.webp
    ├── iowa_confidence_map.webp
    ├── iowa_donut_chart.webp
    │
    ├── bergen_street.PNG             # Out-of-distribution: Street photo
    ├── bergen_street_predicted_mask.webp
    ├── bergen_street_confidence_map.webp
    └── bergen_street_donut_chart.webp
```

---

## Datasets

### DeepGlobe Land Cover Classification
- **Source:** [Kaggle — balraj98](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)
- **Size:** 803 satellite images at 2448×2448px with pixel-wise RGB masks
- **Classes:** Urban Land, Agriculture, Rangeland, Forest, Water, Barren Land, Unknown
- **Coverage:** Thailand, Indonesia, India (tropical/subtropical rural terrain)
- **Split:** 85% train (682 images) / 15% validation (121 images), seed 42

### LandCover.ai
- **Source:** [Kaggle — adrianboguszewski](https://www.kaggle.com/datasets/adrianboguszewski/landcoverai)
- **Paper:** Boguszewski, A., Batorski, D., Ziemba-Jankowska, N., Dziedzic, T., Zambrzycka, A. *LandCover.ai: Dataset for Automatic Mapping of Buildings, Woodlands, Water and Roads from Aerial Imagery.* CVPR Workshops, 2021. [arXiv:2005.02264](https://arxiv.org/abs/2005.02264)
- **Size:** 41 large orthophoto tiles covering 216 sq km of rural Poland at 25–50 cm/px
- **Classes:** Background, Building, Woodland, Water
- **Split:** 85% train / 15% validation, seed 42; large tiles are randomly cropped to 512×512 patches during loading

Dataset files are not included in this repository. Add both via the Kaggle UI when running the notebook.

---

## Architecture and Training

```
Input image (512 x 512 x 3)
        |
   ResNet-50 Encoder       pretrained on ImageNet, lr = 3e-5
        |
   U-Net Decoder           skip connections at 4 scales, lr = 3e-4
        |
   Segmentation Head       N output channels (7 for DeepGlobe, 4 for LandCover.ai)
```

| Setting | Value |
|---|---|
| Library | segmentation-models-pytorch 0.3.3 |
| Loss | CrossEntropy (0.5) + Dice (0.5), inverse-frequency class weights |
| Optimizer | AdamW, discriminative LR (encoder 10x lower than decoder) |
| Scheduler | CosineAnnealingWarmRestarts (T0=10, T_mult=2) |
| Batch size | 16 (8 per GPU), gradient accumulation x2, effective batch 32 |
| Epochs | 30 |
| Precision | Mixed precision (AMP) |
| Hardware | 2x NVIDIA Tesla T4 (15GB each) via Kaggle, DataParallel |
| Random seed | 42 (fixed throughout) |

---

## Training Progression

DeepGlobe mIoU improved from 0.226 at epoch 1 to 0.637 at epoch 29, with steady convergence and no significant overfitting gap between train and validation loss.

LandCover.ai mIoU improved from 0.424 at epoch 1 to 0.798 at epoch 27. The higher ceiling compared to DeepGlobe reflects the simpler 4-class problem and cleaner class boundaries in the orthophoto data.

---

## Error Analysis

Three-tier inference testing was performed to evaluate model behaviour across input types:

| Test | Input Type | Mean Confidence | Low-Conf Pixels | Key Finding |
|---|---|---|---|---|
| In-distribution | DeepGlobe farmland | 0.745 | 3.4% | Correct spatially coherent prediction. Agriculture (50.1%) and Water (9.1%) cleanest. Small Urban Land cluster (1.3%) correctly detected. |
| Distribution shift | Google Earth suburban aerial (USA) | 0.631 | 9.4% | Confidence drops measurably. Dense suburban tree canopy collapses into Rangeland (51%) rather than Forest — fragmented canopy texture does not match DeepGlobe's training distribution. Urban Land rises to 21.6%. |
| Out-of-distribution | Ground-level street photo | 0.905 | 1.6% | Overconfident wrong prediction: 95.9% Urban Land. Classic softmax overconfidence failure on OOD input. |

The OOD overconfidence result (high confidence on clearly wrong input) is a known limitation of softmax-based confidence as an OOD detection mechanism. The model cannot express "I don't recognise this input" — it always assigns a class. Future work could address this using feature-space distance metrics or an explicit rejection class.

---

## Interactive Demo

Section 17 of the notebook launches a Gradio web interface:

```python
demo.launch(share=True)
```

Features include: image upload with class visibility filter, predicted mask overlay, donut chart of class distribution with percentages, confidence heatmap, and an OOD detection badge (In-distribution / Uncertain / Likely out-of-distribution) based on mean confidence and low-confidence pixel fraction.

---

## Reproduction Steps

1. Upload `land_cover_classification.ipynb` to a new Kaggle notebook
2. Add both datasets via **+ Add Data**:
   - `balraj98/deepglobe-land-cover-classification-dataset`
   - `adrianboguszewski/landcoverai`
3. Set accelerator to **GPU T4 x2** under Settings
4. Run all cells — the first cell installs dependencies automatically
5. Training takes approximately 2–3 hours on dual T4

**Resuming after a session disconnect:** The training loop (Section 10) saves `latest_model.pth` after every epoch. If the session restarts but `/kaggle/working` is still intact, re-running the training cell will automatically detect this file and resume from the last completed epoch. See Section 2b in the notebook for full instructions on both reconnect scenarios.

---

## Limitations

- Trained on tropical/subtropical satellite imagery (DeepGlobe) and Polish orthophotos (LandCover.ai) — generalisation to other regions and sensor types is limited
- Rangeland class is persistently weak (IoU 0.348) due to visual overlap with Agriculture and Barren Land; this appears to be a dataset-level challenge rather than purely a model capacity issue
- No explicit OOD rejection mechanism — the model will produce confident predictions on arbitrary input regardless of whether it resembles the training distribution
- LandCover.ai validation metrics show higher variance across epochs than DeepGlobe, likely due to the small number of source tiles (41 total) making the validation split highly sensitive to which specific tiles are held out

---

## References

1. Demir, I. et al. *DeepGlobe 2018: A Challenge to Parse the Earth through Satellite Images.* CVPR Workshops, 2018.
2. Boguszewski, A. et al. *LandCover.ai: Dataset for Automatic Mapping of Buildings, Woodlands, Water and Roads from Aerial Imagery.* CVPR Workshops, 2021. [arXiv:2005.02264](https://arxiv.org/abs/2005.02264)
3. Ronneberger, O. et al. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI, 2015.
4. He, K. et al. *Deep Residual Learning for Image Recognition.* CVPR, 2016.
5. Yakubovskiy, P. *Segmentation Models PyTorch.* GitHub, 2020. https://github.com/qubvel/segmentation_models.pytorch

---

## License

Code in this repository is released under the MIT License.
DeepGlobe dataset: see [original challenge terms](https://competitions.codalab.org/competitions/18468).
LandCover.ai dataset: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
