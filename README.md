> 🚧 The GeoFloodNet online flood monitoring platform is under development and will be released soon.

# GeoFloodNet

## Toward Rapid Flood Mapping Anywhere via Terrain- and Land-Cover-Conditioned Optical–SAR Fusion

This repository serves as the official project page for **GeoFloodNet**, a geographically conditioned cross-modal framework for rapid event-induced flood inundation mapping.

GeoFloodNet is built on the idea that **flood evidence is not geographically invariant**. Similar optical–SAR discrepancies may indicate true inundation in low-lying cropland, but may also correspond to wetlands, permanent water, terrain shadow, smooth artificial surfaces, or other flood-like background conditions elsewhere. GeoFloodNet addresses this challenge by interpreting pre-event optical context and post-event SAR evidence under local terrain and land-cover conditions.

An online flood monitoring platform powered by GeoFloodNet is currently under development and will be released soon. The platform will allow users to upload an area of interest, select a monitoring period, browse available Sentinel-1 and Sentinel-2 observations, run flood extraction online, and download the generated inundation results.

<p align="center">
  <img src="ai-geoflood.png" width="850" alt="GeoFloodNet value illustration">
</p>

---

## Overview

Rapid flood mapping requires not only timely satellite observations, but also reliable interpretation across heterogeneous geographic environments. In practical emergency response, pre-event optical imagery can provide land-surface context before a disaster, while post-event SAR imagery provides all-weather crisis-time observations during or immediately after the flood event.

However, optical–SAR discrepancies do not have fixed flood semantics across regions. A low-backscatter SAR response may indicate newly inundated cropland in one area, but may correspond to wetland conditions, permanent water, smooth surfaces, or terrain-related effects in another. This makes rapid flood mapping a geographically conditioned inference problem rather than a simple cross-modal change detection task.

This project introduces:

* **GeoFloodNet**, a geographically conditioned optical–SAR fusion framework for rapid event-induced inundation mapping;
* **GeoFlood-275**, a global event-level benchmark constructed to develop and evaluate geographically conditioned flood mapping;
* an upcoming **online flood monitoring platform** that will provide an accessible interface for applying GeoFloodNet to user-defined areas of interest.

---

## Online Flood Monitoring Platform

The GeoFloodNet online platform is designed to make rapid flood mapping accessible without requiring users to manually download, preprocess, and align multi-source remote-sensing data.

The platform will support a streamlined workflow:

| Step                         | Description                                                                                                           |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| 1. Upload AOI                | Upload a target area using GeoJSON or Shapefile.                                                                      |
| 2. Select time range         | Specify the monitoring period for the target flood event.                                                             |
| 3. Retrieve observations     | The platform automatically searches available Sentinel-1 and Sentinel-2 data within the selected area and time range. |
| 4. Prepare geographic priors | Terrain and land-cover information are automatically prepared in the background.                                      |
| 5. Select image pair         | Users select the appropriate pre-event optical and post-event SAR observations from the available candidates.         |
| 6. Run flood extraction      | GeoFloodNet performs geographically conditioned optical–SAR inference online.                                         |
| 7. Download results          | Users can download the generated flood inundation map and associated outputs.                                         |

The platform is currently being finalized and will be made publicly accessible soon.

---

## Key Idea

### From cross-modal difference detection to flood-evidence interpretation

Most cross-modal flood mapping methods treat differences between pre-event optical imagery and post-event SAR imagery as direct flood cues. GeoFloodNet instead asks:

> Is the observed optical–SAR evidence flood-relevant under the local terrain and land-cover context?

This distinction is important because flood-like image responses may arise from multiple non-flood conditions, including permanent water, wetlands, wet soils, agricultural activities, smooth built-up surfaces, and terrain effects.

### Geographic conditioning as a core mechanism

GeoFloodNet does not simply concatenate geographic priors as additional input channels. Instead, terrain and land-cover priors are used to condition the interpretation of optical–SAR evidence.

Specifically, geographic priors are used to:

* modulate modality-specific optical and SAR features;
* estimate local modality reliability;
* regulate cross-modal fusion;
* suppress flood-like background responses in complex environments.

This design enables GeoFloodNet to assign different flood semantics to similar optical–SAR patterns across different geographic settings.

---

## GeoFloodNet

GeoFloodNet is a geographically conditioned optical–SAR fusion framework for rapid event-induced flood inundation mapping.

The model uses:

| Input                                | Role                                                                           |
| ------------------------------------ | ------------------------------------------------------------------------------ |
| Pre-event Sentinel-2 optical imagery | Provides land-surface semantics and background context before the flood event. |
| Post-event Sentinel-1 SAR imagery    | Provides all-weather crisis-time evidence of flood-related surface responses.  |
| Slope                                | Provides terrain constraints for flood plausibility.                           |
| ESA WorldCover                       | Provides stable global land-cover semantics.                                   |
| Dynamic World NRT                    | Provides event-adjacent land-cover context.                                    |

GeoFloodNet interprets flood evidence through geographic conditioning. Instead of treating optical–SAR differences as universally meaningful change signals, it evaluates whether such differences are consistent with event-induced inundation under the local environmental context.

---

## GeoFlood-275

GeoFlood-275 is a global event-level benchmark constructed for geographically conditioned rapid flood inundation mapping.

Each event–AOI pair contains:

| Component                       | Description                           |
| ------------------------------- | ------------------------------------- |
| Pre-event optical image         | Sentinel-2 RGB + NIR                  |
| Post-event SAR image            | Sentinel-1 VV + VH                    |
| Reference map                   | EMSR-derived event-induced inundation |
| Terrain prior                   | Slope derived from Copernicus DEM     |
| Stable land-cover prior         | ESA v200 WorldCover                   |
| Event-adjacent land-cover prior | Dynamic World NRT                     |
| Spatial resolution              | 10 m                                  |
| Patch size                      | 512 × 512                             |

Dataset statistics:

| Split      | Events | Samples | Protocol                     |
| ---------- | -----: | ------: | ---------------------------- |
| Train      |    234 | 125,552 | Historical flood events      |
| Validation |     20 |   4,476 | Temporally separated events  |
| Test       |     21 |   3,527 | Temporally separated events  |
| Total      |    275 | 133,555 | Global event-level benchmark |

The reference labels correspond to **event-induced inundation**, not generic surface water. Permanent and background water bodies are excluded under the adopted reference definition.

---

## Code Release

This repository now includes the official GeoFloodNet training and evaluation code for research reproducibility.

### Repository Layout

```text
network/
  GeoFloodNet.py        # GeoFloodNet model
utils/
  dataloader.py         # main GeoFlood-275 GeoTIFF dataloader
  dataload_v2.py        # optional extended dataloader for extra products
  utils.py              # training helpers
train.py                # training entry point
test.py                 # validation/test evaluation and visualization
```

### Dataset Layout

`train.py` and `test.py` expect the GeoTIFF dataset root to contain `Train`, `Val`, and `Test` folders:

```text
Benchmark_all/
  Train/
    Sentinel-2/
    Sentinel-1/
    CD_Flood/
    DEM/
    ESAV200/
    Dynamic_Landcover/
    Slope_Norm/
  Val/
    ...
  Test/
    ...
```

The benchmark dataset is publicly hosted on Hugging Face:

https://huggingface.co/datasets/jiepanli/GeoFlood-275

The released GeoTIFF files are not pre-normalized training tensors. `utils/dataloader.py` applies normalization at load time: Sentinel-1 is clipped to `[-45, 25]` dB and mapped to `[-1, 1]`, Sentinel-2 is clipped to `[0, 6000]` and mapped to `[-1, 1]`, DEM is scaled by `/5000` and standardized, slope is clipped to `[0, 1]`, ESA WorldCover classes are remapped to contiguous IDs, Dynamic World labels remain integer labels, and flood masks are mapped to `{0, 1}` when needed.

### Installation

Install PyTorch for your CUDA version from the official PyTorch instructions, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

### Training

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset_root /path/to/GeoFlood-275/Benchmark_all \
  --save_path ./Experiments/GeoFloodNet \
  --batchsize 8 \
  --amp
```

### Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --model floodnet \
  --dataset_root /path/to/GeoFlood-275/Benchmark_all \
  --load ./Experiments/GeoFloodNet/Val_best.pth \
  --batchsize 4 \
  --amp
```

Evaluation writes per-split metrics, confusion matrices, prediction masks, and visualizations under the checkpoint directory unless `--output_dir` is specified.

---

## Platform Availability

The GeoFloodNet online flood monitoring platform is coming soon.

Once released, users will be able to:

* upload GeoJSON or Shapefile AOIs;
* select flood monitoring time ranges;
* browse available Sentinel-1 and Sentinel-2 observations;
* run GeoFloodNet-based flood extraction online;
* visualize and download flood inundation results.

No local installation or manual multi-source data preparation will be required.

---

## Notes

The manuscript associated with this project has been submitted for peer review. The official GeoFloodNet source code is released in this repository, and the GeoFlood-275 benchmark dataset is released on Hugging Face for research reproducibility:

https://huggingface.co/datasets/jiepanli/GeoFlood-275

---

## Citation

```bibtex
@misc{li2026geoflood275,
  title = {Toward Rapid Flood Mapping Anywhere via Terrain- and Land-Cover-Conditioned Optical-SAR Fusion},
  author = {Li, Jiepan and Huang, He and Li, Wenke and Li, Linxin and Xie, Anqi and Ye, Ruoru and Hu, Lei and Hu, Ting and He, Wei and Zhang, Liangpei},
  year = {2026},
  note = {Manuscript under revision for Remote Sensing of Environment}
}
```

---

## Contact

For questions, collaborations, or platform updates, please open an issue in this repository.
