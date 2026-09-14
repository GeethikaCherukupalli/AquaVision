# AquaVision — Project Context

> **Purpose:** This document is the handoff/context file for Claude Code working in the new AquaVision production repository.
>
> It describes work, decisions, experiments, constraints, and validated facts from the earlier AquaVision development repository.
>
> **Important:** This is a project-context document, not the production implementation specification. The production architecture and implementation requirements are described separately in `AquaVision_Claude_Code_Production_Spec.md`.

---

# 1. Project Identity

## Project

**AquaVision**

## Smart India Hackathon

- **SIH:** Smart India Hackathon 2026
- **Problem Statement:** **PS 26143**
- **Organization:** NTRO
- **Category:** Software
- **Theme:** Disaster Management
- **Problem statement title:**

> “Leveraging satellite imagery to determine Oil spills at sea along with AIS data correlations to identify vessel responsible for the spill”

- **Submission deadline:** 30 September 2026
- **Current project status:** The team's internal SIH evaluation has been completed and the team has passed. The current objective is to move from an internal/prototype implementation toward a credible deployment-stage demonstration.

---

# 2. Core Problem We Are Solving

AquaVision is intended to help emergency response teams and oil-spill cleanup/response crews detect potential offshore oil spills, understand their physical characteristics, estimate where and when the spill likely originated, predict its movement, and correlate that information with vessel movements to identify vessels of interest.

The intended high-level pipeline is:

```text
SATELLITE OBSERVATION
        ↓
POTENTIAL OIL-SPILL DETECTION
        ↓
SPILL SEGMENTATION
        ↓
PHYSICAL CHARACTERIZATION
        ↓
OCEAN + WEATHER DATA
        ↓
OPENDRIFT HINDCAST
        ↓
ORIGIN PROBABILITY REGION + TIME WINDOW
        ↓
OPENDRIFT FORECAST
        ↓
AIS CORRELATION
        ↓
VESSEL-OF-INTEREST RANKING
        ↓
RESPONSE / DECISION SUPPORT
```

The conceptual product flow is:

**SEE → RECONSTRUCT → TRACE → ATTRIBUTE → RESPOND**

---

# 3. Critical Scientific Positioning

These rules are non-negotiable.

## 3.1 SAR does NOT prove that a dark patch is oil

Sentinel-1 SAR imagery can reveal dark features that may be consistent with oil spills, but dark SAR signatures can also be caused by lookalikes such as:

- low-wind areas
- biogenic films
- natural sea-surface phenomena
- vessel wakes
- other SAR-dark features

Therefore the system should use terminology such as:

- **potential oil spill**
- **oil-spill candidate**
- **oil-like signature**
- **candidate spill region**

Do NOT present SAR classification as laboratory-level confirmation of oil.

---

## 3.2 AIS attribution does NOT prove vessel guilt

The vessel-ranking stage is intended to identify:

- vessels of interest
- high-correlation vessels
- candidate responsible vessels
- vessels requiring investigation

It must NOT state that a vessel is definitely guilty based only on the system's correlation.

The ranking should be presented as an evidence/correlation score based on measurable factors such as:

- spatial proximity
- temporal overlap
- trajectory relationship
- behavioral anomalies
- other configurable evidence

The final wording should clearly distinguish **correlation/evidence** from **proof or legal attribution**.

---

## 3.3 Origin location is probabilistic

OpenDrift hindcast should not be treated as a magical exact-origin calculator.

The intended output is:

```text
Origin probability region
+
Representative/weighted origin point
+
Estimated origin time window
+
Confidence
+
Uncertainty
```

The system should communicate uncertainty rather than pretending to know an exact origin point.

---

# 4. Product Modes

AquaVision is intended to support two major modes.

## 4.1 Historical Analysis Mode

User supplies/selects a historical Sentinel-1 acquisition or relevant scene.

Pipeline:

```text
Historical Sentinel-1 product
        ↓
Stage 1
        ↓
Spill characterization
        ↓
Environmental data
        ↓
OpenDrift hindcast
        ↓
Likely origin region/time
        ↓
Historical AIS retrieval
        ↓
Vessel correlation/ranking
        ↓
Historical incident reconstruction
```

This mode is especially important for a reliable demo because historical satellite scenes, environmental data, and AIS data can be reproduced.

---

## 4.2 Monitoring / Near-Real-Time Mode

The intended operational concept is:

```text
Copernicus Data Space Ecosystem
        ↓
New Sentinel-1 acquisition/product
        ↓
AOI/time filtering
        ↓
Stage 1
        ↓
If candidate:
    Stage 2 → Stage 3
```

Use **near-real-time monitoring**, not "continuous real-time satellite monitoring."

Sentinel-1 does not provide continuous video-like coverage of the ocean.

The monitoring layer should detect/process newly available Sentinel-1 products rather than pretending that imagery is available continuously.

---

# 5. Ownership / Team Division

The user is primarily responsible for the **backend**.

A teammate is responsible for the **frontend**.

Frontend-specific implementation/integration instructions should be explicitly labelled:

**[FRONTEND NOTE]**

Do not unnecessarily redesign or take over the frontend while implementing the backend.

The backend should expose clean APIs/contracts so the frontend teammate can consume them.

---

# 6. Current Development Strategy

This repository is intentionally a **fresh production repository**.

The previous AquaVision repository contained experimental/prototype code, hardcoded demonstrations, notebooks, and development shortcuts.

Do NOT blindly copy the previous implementation into this repository.

Instead:

- carry forward validated decisions
- carry forward useful model architecture choices
- carry forward dataset knowledge
- carry forward measured experimental results
- redesign the codebase cleanly
- avoid hardcoded demo behavior
- make providers modular
- make model versions explicit
- make scientific outputs reproducible

The old prototype should be treated as experimental evidence, not as production architecture.

---

# 7. Stage 1 — Current ML Concept

Stage 1 is currently the most mature technical component.

The intended sequence is:

```text
Sentinel-1 SAR
      ↓
Preprocessing
      ↓
512 × 512 patches
      ↓
ResNet18 binary gate
      ↓
Potential oil patch?
      ↓ YES
U-Net segmentation
      ↓
Pixel-level spill mask
      ↓
Scene-level reconstruction
      ↓
Physical characterization
```

The ResNet and U-Net perform different jobs.

---

# 8. ResNet18 — Current Design

## Purpose

The ResNet18 is a **binary candidate gate**.

It answers:

> "Should this spatial patch be passed to the more expensive/precise U-Net segmentation stage?"

It is NOT intended to replace segmentation.

---

## 8.1 Classes

The current intended classification is binary:

```text
0 = NOT OIL
1 = OIL
```

For training purposes:

```text
No-oil + Lookalike → NOT OIL
Oil → OIL
```

This is intentionally not a 3-class production classifier.

The lookalike class is used as a hard negative.

---

## 8.2 Input

Sentinel-1 SAR input:

```text
2 channels
    Channel 0 → VV
    Channel 1 → VH

Patch size:
512 × 512
```

The model therefore receives:

```text
[2, 512, 512]
```

rather than RGB.

---

## 8.3 ResNet18 Architecture

Base model:

```text
torchvision.models.resnet18
```

The original first convolution expects 3 channels.

It was changed to:

```text
2 input channels
```

The previous experiment initialized the new convolution from the pretrained RGB convolution by averaging the original RGB weights and repeating the averaged weights across the two SAR channels.

Conceptually:

```text
Original:
Conv2d(3 → 64)

New:
Conv2d(2 → 64)
```

The final classification layer was changed to:

```text
Linear(512, 1)
```

The output is a single logit.

Use:

```text
sigmoid(logit) → P(oil)
```

during inference.

---

# 9. Patch Label Generation

This is an important previous decision.

The original scenes are 2048 × 2048.

They are tiled into:

```text
512 × 512
```

giving:

```text
4 × 4 = 16 patches per scene
```

The ResNet patch label is derived from the corresponding segmentation mask.

For each patch:

```text
oil_fraction =
    number of oil pixels
    --------------------
    total pixels
```

Initial working threshold:

```text
oil_fraction >= 0.01
```

meaning:

```text
>= 1% oil pixels → OIL
< 1% oil pixels  → NOT OIL
```

This 1% threshold is an **initial engineering threshold**, not a scientifically final value.

It should remain configurable and should be revisited on the full dataset.

---

# 10. Why Patch-Level Labeling Matters

Do NOT label every 512 × 512 patch from an oil-containing scene as "oil."

An oil scene can contain large areas of ocean with no spill.

The correct approach is:

```text
Scene
 ↓
Mask
 ↓
512 × 512 patches
 ↓
Calculate oil fraction for each patch
 ↓
Assign patch labels
```

This produces spatially meaningful training examples for the ResNet gate.

---

# 11. Previous Subset Experiment

A small development subset was used to validate the pipeline before moving to the full dataset.

Subset structure previously contained:

```text
Oil/
No_oil/
look_alike/
Oil_Mask/
```

with approximately:

```text
20 Oil images
20 No-oil images
20 Lookalike images
20 Oil masks
```

Each image:

```text
2048 × 2048 × 2
```

Each mask:

```text
2048 × 2048 × 1
```

---

# 12. Mask Validation Results

A representative oil mask was checked.

Properties:

```text
Width: 2048
Height: 2048
Bands: 1
dtype: uint8
Minimum: 0
Maximum: 1
Unique values: [0, 1]
```

The mask therefore acts as a binary segmentation mask.

Example:

```text
oil pixels = 14,539
total pixels = 4,194,304
oil fraction ≈ 0.003466
```

That is approximately:

```text
0.3466%
```

of the full scene.

---

# 13. Image Validation Results

Representative source images were checked and showed:

```text
shape = (2, 2048, 2048)
dtype = float32
bands = 2
CRS = EPSG:4326
```

The two channels correspond to the SAR VV/VH representation.

Representative statistics from the subset included:

### Oil

```text
min  ≈ -54.12
max  ≈   5.55
mean ≈ -27.53
std  ≈   7.65
```

### No-oil

```text
min  ≈ -54.03
max  ≈ -15.44
mean ≈ -29.31
std  ≈   7.78
```

### Lookalike

```text
min  ≈ -54.47
max  ≈ -12.10
mean ≈ -31.76
std  ≈   4.41
```

These are dataset observations, not universal Sentinel-1 thresholds.

---

# 14. Patch Distribution Experiment

Using the 20-oil-scene subset:

```text
20 scenes × 16 patches
= 320 patches
```

Results:

```text
Non-zero oil patches: 118
Empty patches:        202
```

Oil fraction statistics:

```text
mean ≈ 0.017287
std  ≈ 0.038117
min  = 0
median = 0
max ≈ 0.240284
```

Threshold experiments:

```text
Threshold     Positive patches
--------------------------------
0.10%         100
0.25%          96
0.50%           91
1.00%           81
2.00%           70
5.00%           45
```

The initial working threshold was therefore selected as:

```text
1%
```

Again, this is a configurable starting point.

---

# 15. Combined Subset Index

A small index was constructed using:

```text
Oil
No_oil
Lookalike
```

with:

```text
320 patches/class
960 total patches
```

Patch-level label distribution:

```text
NOT OIL = 879
OIL     = 81
```

Within Oil scenes:

```text
NOT OIL = 239
OIL     = 81
```

This illustrates why scene-level class labels cannot simply be copied onto every patch.

---

# 16. Train/Validation Split

The split was done **by scene**, not by random patch.

This is important because randomly splitting patches from the same scene can create spatial leakage.

Previous split:

```text
TRAIN:
16 scenes
768 patches
256 patches/class

Labels:
NOT OIL = 701
OIL     = 67
```

```text
VALIDATION:
4 scenes
192 patches
64 patches/class

Labels:
NOT OIL = 178
OIL     = 14
```

Scene overlap:

```text
none
```

Therefore the development split avoided scene-level leakage.

---

# 17. ResNet Training Configuration

Previous experiment used:

```text
Model:
ResNet18

Input:
2 × 512 × 512

Loss:
BCEWithLogitsLoss

Positive class weighting:
701 / 67 ≈ 10.46

Optimizer:
AdamW

Learning rate:
1e-4

Weight decay:
1e-4

Batch size:
8

Scheduler:
ReduceLROnPlateau

Scheduler factor:
0.5

Scheduler patience:
2

Epochs:
10
```

---

# 18. SAR Normalization

Training-only channel statistics from the subset experiment were:

```text
VV mean = -32.509415 dB
VV std  =   4.573598 dB

VH mean = -21.745690 dB
VH std  =   5.040430 dB
```

Normalization was channel-wise:

```text
normalized = (value - mean) / std
```

The production implementation must calculate normalization statistics from the appropriate training data and store them with the model metadata.

Do not blindly assume the subset statistics are the final full-dataset statistics.

Do not use ImageNet normalization for the SAR channels.

---

# 19. Previous ResNet Results

The 10-epoch subset experiment ended with:

```text
Epoch 10/10

Train Loss:
0.0825

Validation Loss:
0.9171

Precision:
0.875

Recall:
0.500

F1:
0.636

PR-AUC:
0.804
```

Confusion matrix:

```text
[[177, 1],
 [  7, 7]]
```

Therefore:

```text
TN = 177
FP = 1
FN = 7
TP = 7
```

---

# 20. Interpretation of ResNet Results

The model learned a useful signal on the small development subset.

However:

```text
Recall = 50%
```

at the default 0.5 decision threshold is not acceptable for a production candidate gate.

A false negative at this stage is especially problematic because:

```text
ResNet says NOT OIL
        ↓
U-Net never sees the patch
        ↓
potential spill is missed
```

Therefore the final operating threshold should prioritize **recall**, subject to acceptable false-positive workload.

The subset is too small to claim production performance.

In particular, there were only:

```text
14 positive validation patches
```

so one validation example changes recall by roughly 7.14 percentage points.

The subset experiment should be treated as a **pipeline validation**, not final model validation.

---

# 21. Threshold Sweep From Development Experiment

Previous threshold sweep approximately produced:

```text
Target recall       Threshold     Precision
--------------------------------------------
>= 60%              0.140         0.846
>= 70%              0.140         0.846
>= 80%              0.110         0.800
>= 90%              0.050         0.295
>= 95%              0.032         0.206
```

These numbers are from the tiny development validation set.

They MUST NOT be presented as final production threshold performance.

An initial candidate operating range may be around:

```text
0.11–0.14
```

but the final threshold must be selected after training/validation on the full appropriate dataset.

The threshold should be stored in model metadata/configuration rather than hardcoded into application logic.

---

# 22. Important ResNet Production Goal

The next real training experiment is on the **full available training data**, not the tiny subset.

The production process should:

1. enumerate source scenes
2. derive patch labels from masks
3. split scenes into train/validation
4. calculate training-only normalization statistics
5. train ResNet18
6. evaluate PR-AUC
7. evaluate recall/precision/F1
8. inspect threshold curves
9. choose an operating threshold
10. save model weights
11. save model metadata
12. record dataset version and training configuration

The test/held-out dataset should remain isolated.

---

# 23. U-Net — Current Design

The U-Net is responsible for **pixel-level segmentation** after a patch has passed the candidate gate.

Previous U-Net architecture used:

```text
Input:
2 channels

Spatial size:
512 × 512

Encoder:
2 → 32 → 64 → 128

Bottleneck:
128 → 256

Decoder:
skip connections

Output:
1 × 512 × 512
```

Loss:

```text
BCE + Dice
```

Optimizer:

```text
AdamW
```

Learning rate:

```text
1e-4
```

---

# 24. Previous U-Net Results

A previous prototype reached approximately:

```text
Epoch 07/10

Dice ≈ 0.8536
IoU  ≈ 0.7512
```

These are **prototype/subset results**.

They must not be described as final full-dataset production metrics.

The old U-Net checkpoint/demo implementation was considered hardcoded prototype work and should not simply be copied into the new repository.

---

# 25. Stage 1 U-Net Output

The U-Net output should ultimately provide:

```text
pixel-level probability/mask
```

which can then be thresholded and postprocessed into:

```text
binary spill mask
```

The system should reconstruct the scene from the processed 512 × 512 candidate patches.

Expected Stage 1 output should include:

```text
Potential spill mask
Spill polygon(s)
Area
Perimeter
Centroid
Bounding box
Length
Width
Orientation
Compactness
Confidence / model metadata
Satellite acquisition metadata
```

The exact physical-characterization calculations belong to the Stage 1 postprocessing/characterization layer.

---

# 26. Patch-to-Scene Reconstruction

The source scene is:

```text
2048 × 2048
```

and is divided into:

```text
16 × 512 × 512 patches
```

The ResNet identifies candidate patches.

Only candidate patches need to be passed to the U-Net.

The U-Net masks are then placed back into their original spatial positions.

Conceptually:

```text
2048 × 2048 scene
┌────────┬────────┬────────┬────────┐
│ patch  │ patch  │ patch  │ patch  │
├────────┼────────┼────────┼────────┤
│ patch  │ patch  │ patch  │ patch  │
├────────┼────────┼────────┼────────┤
│ patch  │ patch  │ patch  │ patch  │
├────────┼────────┼────────┼────────┤
│ patch  │ patch  │ patch  │ patch  │
└────────┴────────┴────────┴────────┘
```

The scene-level segmentation must preserve the original row/column offsets.

---

# 27. Physical Characterization

Once the spill mask is reconstructed, calculate useful geometric properties.

Potential outputs:

```text
area
perimeter
centroid
bounding box
length
width
orientation
compactness
```

Important:

- Use the source georeferencing.
- Avoid treating raw pixel counts as square meters without conversion.
- Use a suitable geographic/projected coordinate transformation or geodesic calculation.
- Preserve the original pixel mask and vector geometry.
- Record the methodology used to calculate physical measurements.

---

# 28. Full Dataset Information

The SIH-provided Zenodo dataset is split into three parts.

## Part I

Zenodo record:

```text
8346860
```

Contains approximately:

```text
1200 oil-spill images
1200 masks
```

Image characteristics:

```text
TIFF
2048 × 2048
2 channels
VV/VH
```

Only the images are georeferenced.

Masks are treated as segmentation matrices and may not contain georeferencing.

---

## Part II

Zenodo record:

```text
8253899
```

Contains approximately:

```text
685 no-oil images
685 no-oil masks

685 lookalike images
685 lookalike masks
```

Images are:

```text
2048 × 2048 × 2 TIFF
```

For the no-oil and lookalike examples, the masks represent no oil target.

---

## Part III

Zenodo record:

```text
13761290
```

Held-out test data:

```text
150 lookalike
150 no-oil
150 oil
```

with corresponding masks.

**Do not randomly mix Part III into training or validation.**

It should remain held out for final evaluation.

---

# 29. Current Google Drive Dataset Location

The full dataset is stored in Google Drive.

Current known root:

```text
/content/drive/MyDrive/oil_spill
```

Expected top-level structure:

```text
oil_spill/
├── Part1/
├── Part2/
└── Part3/
```

The exact nested folder names inside Part1/Part2/Part3 must be inspected rather than guessed.

The dataset is approximately:

```text
90 GB
```

The user has substantial Google Drive storage available.

---

# 30. Colab Data Access

The previous workflow mounted Drive in Google Colab using:

```python
from google.colab import drive
drive.mount('/content/drive')
```

The full dataset should remain in Drive for training.

Do NOT copy the entire ~90 GB dataset into the repository.

Do NOT design the production repository around storing the training dataset.

---

# 31. Efficient Dataset Loading

The previous implementation deliberately avoided creating thousands of physical patch files.

Instead, a patch index stores metadata such as:

```text
image_path
row
col
label
oil_fraction
split
```

and the dataset class uses rasterio windows to lazily read:

```text
512 × 512
```

regions from the original TIFF.

This is the preferred approach for the training pipeline.

Avoid:

```text
90 GB source data
→ thousands/millions of copied patch files
```

when a lazy-window approach is sufficient.

---

# 32. Rasterio Observations

The source TIFFs produced warnings similar to:

```text
TIFFReadDirectory:
Sum of Photometric type-related color channels and ExtraSamples
doesn't match SamplesPerPixel.
Defining non-color channels as ExtraSamples.
```

Rasterio still successfully read the source images as:

```text
2-channel float32
```

This was not treated as a blocker.

Masks also produced:

```text
NotGeoreferencedWarning
```

because the masks do not contain normal georeferencing.

That is expected for the mask matrices and should not automatically be treated as an error.

---

# 33. Previous Dataset Class Concept

The earlier lazy-loading dataset effectively did:

```text
index row
 ↓
image path
 ↓
row/column window
 ↓
rasterio.open()
 ↓
read 512×512 VV/VH
 ↓
float32
 ↓
NaN/Inf cleanup
 ↓
channel normalization
 ↓
torch tensor
```

The production implementation can improve this, but should preserve the underlying concept.

---

# 34. Important Data Leakage Rule

Splitting must happen at **scene level** before patch-level training.

Do NOT:

```text
one scene
 ↓
16 patches
 ↓
randomly put some patches in train
 ↓
remaining patches in validation
```

Instead:

```text
scene IDs
 ↓
train/validation/test scene split
 ↓
patches generated/read within each split
```

This prevents spatially adjacent patches from the same scene appearing across train and validation.

---

# 35. Actual Sentinel-1 Production Inference

Training data and production Sentinel-1 imagery may not arrive in exactly the same representation.

The eventual production ingestion path must ensure that inference imagery is comparable to the training representation.

The intended representation is:

```text
Sigma0 / backscatter in dB
VV
VH
```

Raw Sentinel-1 GRD ingestion may require preprocessing/calibration.

Potential tooling includes:

- ESA SNAP
- GDAL
- pyroSAR
- s1ard
- equivalent validated Sentinel-1 preprocessing

Do not assume that a raw downloaded Sentinel-1 product can simply be fed into the trained model.

The exact preprocessing chain must be validated against the training data.

---

# 36. Stage 2 — Environmental Reconstruction

After Stage 1 produces a spill geometry, the system needs environmental forcing.

Primary inputs:

```text
surface ocean currents
wind
```

Potential additional variables:

```text
waves
sea-surface temperature
temperature
diffusivity
```

The most important variables for the first working version are:

```text
ocean current U/V
wind U/V
```

---

# 37. Planned Ocean Current Provider

The planned primary ocean-current source is:

**Copernicus Marine**

The integration should be abstracted behind a provider interface.

Concept:

```text
OceanCurrentProvider
        ↓
Copernicus Marine implementation
```

The provider should return canonical data such as:

```text
time
latitude
longitude
u
v
```

with clearly defined units, ideally:

```text
m/s
```

Provider credentials must never be hardcoded.

---

# 38. Planned Weather/Wind Provider

The first intended weather provider is:

**Open-Meteo**

It can provide wind information and historical/forecast weather depending on endpoint/model.

However, weather access should also be abstracted.

Concept:

```text
WeatherProvider
        ↓
Open-Meteo implementation
```

The canonical internal representation should contain:

```text
wind_u
wind_v
wind_speed
wind_direction
timestamp
latitude
longitude
```

A future ECMWF/ERA5-compatible source can be added without changing the rest of the pipeline.

---

# 39. Environmental Data Provenance

Every simulation should record:

```text
provider
dataset/model
variable names
time range
spatial extent
retrieval timestamp
units
```

This is important for reproducibility.

Environmental data should be cached when practical.

Do not silently substitute one provider for another while pretending they are equivalent.

---

# 40. OpenDrift

The planned physics engine is:

**OpenDrift**

For oil-specific simulation, the intended module is:

```python
from opendrift.models.openoil import OpenOil
```

OpenDrift can use external readers for:

- currents
- wind
- waves
- other environmental forcing

The implementation should use actual environmental datasets/readers rather than fake trajectories.

---

# 41. OpenDrift Hindcast

The core idea:

```text
Observed spill at time T
        ↓
Run particle simulation backward
        ↓
Possible previous positions
        ↓
Probability/density distribution
        ↓
Likely origin region/time
```

OpenDrift supports backward simulations using a negative time direction.

The hindcast should use an **ensemble**, not a single deterministic particle.

---

# 42. Hindcast Particle Initialization

Do NOT initialize all particles from only the spill centroid.

Better:

```text
segmented spill polygon
        ↓
sample multiple initial points
        ↓
particle ensemble
        ↓
OpenDrift hindcast
```

This preserves uncertainty in the observed spill geometry.

---

# 43. Hindcast Output

The hindcast should generate:

```text
particle trajectories
particle density by time
candidate origin regions
representative origin point
origin time window
uncertainty
confidence
```

The representative origin point can be derived from the highest-density region or weighted particle distribution.

The exact algorithm should be documented and configurable.

---

# 44. Origin Probability Region

The system should identify a high-density connected region from the backward particle distribution.

Conceptually:

```text
particles
 ↓
spatial density
 ↓
density threshold / high-density region
 ↓
connected region
 ↓
weighted centroid
```

Output:

```text
origin_probability_polygon
origin_representative_point
origin_time_window
confidence
uncertainty
```

This is the geographic region used for the next AIS stage.

---

# 45. OpenDrift Forecast

After estimating the likely origin/current spill state, the system should support forward simulation.

Example configurable forecast horizons:

```text
+6 hours
+12 hours
+24 hours
+48 hours
+72 hours
```

These should be configuration values rather than hardcoded assumptions.

Forecast output should support:

```text
particle positions
density maps
forecast polygons/regions
time slider
```

The purpose is response planning:

```text
Where could the spill move next?
```

---

# 46. Stage 2 Conceptual Flow

```text
Stage 1 spill geometry
        ↓
Get ocean currents
        +
Get wind
        ↓
Initialize OpenDrift/OpenOil ensemble
        ↓
Hindcast
        ↓
Origin probability region
        +
Origin time window
        ↓
Forecast
        ↓
Future spill movement
```

---

# 47. Sentinel-1 Data Discovery

The intended satellite data ecosystem is:

**Copernicus Data Space Ecosystem (CDSE)**

The current planned approach is to use:

- OData
- STAC
- product metadata
- subscriptions/notifications where useful

The current CDSE STAC endpoint identified during planning is:

```text
https://stac.dataspace.copernicus.eu/v1/
```

Do not hardcode a single satellite scene.

The system should be capable of searching by:

```text
AOI
time range
Sentinel-1 mission/product
product type
```

---

# 48. Historical Satellite Input

For the first reliable demonstration, historical Sentinel-1 products are acceptable and often preferable.

The historical mode should let the user select:

```text
scene/product
acquisition time
location/AOI
```

and then process that scene through the exact same Stage 1 pipeline used by monitoring.

This avoids creating two different ML implementations.

---

# 49. Monitoring Input

The eventual monitoring worker should:

```text
query CDSE
 ↓
find newly available Sentinel-1 product
 ↓
deduplicate product ID
 ↓
download/access product
 ↓
preprocess
 ↓
run Stage 1
```

The monitoring system should track:

```text
product ID
acquisition time
ingestion time
processing status
analysis ID
```

---

# 50. AIS Stage

AIS is used after the hindcast has estimated:

```text
origin probability region
+
origin time window
```

The AIS search region should be:

```text
origin probability region
+
configurable spatial buffer
```

and the time range should be:

```text
origin time window
+
configurable temporal buffer
```

This is much better than searching every vessel in a huge ocean area.

---

# 51. AIS Provider Architecture

AIS should be abstracted behind an interface.

Concept:

```text
AISProvider
    ├── historical AIS implementation
    └── live AIS implementation
```

The implementation should not tightly couple the scoring logic to a single external provider.

---

# 52. Planned Historical AIS Source

A historical AIS provider compatible with datasets such as **MarineCadastre** can be used for historical analysis.

The exact dataset/provider must be validated during implementation.

Do not claim that a particular historical AIS API is available without testing its current access requirements.

---

# 53. Planned Live AIS Source

A planned live source is:

**AISStream**

The current documented connection is a WebSocket endpoint:

```text
wss://stream.aisstream.io/v0/stream
```

It uses a server-side API key.

The API key must never be exposed to the frontend.

The backend should subscribe/filter by bounding boxes and relevant AIS message types.

---

# 54. AIS Trajectory Processing

Raw AIS points should be converted into usable vessel trajectories.

Potential derived values include:

```text
latitude
longitude
timestamp
SOG
COG
heading
turn rate
acceleration
distance to origin region
```

The processing layer should also identify bad/impossible jumps.

For example, an AIS sequence that implies physically impossible movement should be flagged/filtered rather than blindly used.

---

# 55. Vessel Candidate Filtering

After obtaining the origin region/time window:

```text
AIS data
 ↓
spatial filter
 ↓
temporal filter
 ↓
trajectory reconstruction
 ↓
candidate vessels
```

Only vessels plausibly present in the relevant region/time should enter the ranking stage.

---

# 56. Vessel Ranking

The initial planned scoring model is transparent and configurable.

Proposed weights:

```text
Proximity          35%
Temporal overlap   30%
Trajectory         20%
Behavior            15%
--------------------------------
Total              100%
```

These weights are engineering choices for the prototype.

They are **not scientifically validated universal weights**.

The production implementation should make them configurable and expose the evidence breakdown.

---

# 57. Example Vessel Score

A vessel result should look conceptually like:

```text
Vessel:
<identity>

Overall correlation score:
82 / 100

Evidence:
Proximity:         31 / 35
Temporal overlap:  27 / 30
Trajectory:        15 / 20
Behavior:           9 / 15
```

The UI should explain why a vessel ranked highly.

Avoid displaying only:

```text
Vessel X = 82%
```

without explanation.

---

# 58. Vessel Risk/Interest Categories

The planned visual interpretation is:

```text
GREEN
No meaningful correlation

YELLOW / ORANGE
Some behavioral or spatio-temporal anomaly,
but insufficient evidence

RED
High spatio-temporal correlation / high vessel-of-interest score
```

These colors indicate **investigative priority**, not legal guilt.

---

# 59. Frontend Concept

The desired frontend is an **operational intelligence dashboard**, not a generic AI demo.

Potential sections:

```text
Dashboard
Historical Analysis
Monitoring
Simulation
Vessel Attribution
Reports
System Status
```

The map should be central.

---

# 60. Map Layers

The frontend should eventually be capable of displaying:

```text
Sentinel-1 footprint
SAR imagery
ResNet candidate patches
U-Net segmentation
spill polygon
origin probability region
hindcast particles
hindcast density
forecast particles/density
AIS tracks
candidate vessels
```

---

# 61. Simulation UI

The simulation view should support:

```text
time slider
playback
hindcast/forecast distinction
particle/density visualization
origin region
forecast region
```

A responder should be able to understand:

```text
Where was the spill likely originating?
Where could it move?
Which vessels were present?
```

without needing to inspect raw scientific files.

---

# 62. [FRONTEND NOTE]

The frontend should communicate with the backend through defined APIs.

It should NOT directly access:

```text
Copernicus credentials
Open-Meteo provider internals
Copernicus Marine credentials
AIS API keys
OpenDrift
model files
database
```

The backend owns provider authentication and orchestration.

---

# 63. Backend API Direction

The intended API architecture includes endpoints conceptually similar to:

```text
POST /api/v1/analysis

GET /api/v1/analysis/{analysis_id}

GET /api/v1/simulation/{analysis_id}

GET /api/v1/attribution/{analysis_id}
```

There should also be APIs for:

```text
monitoring
jobs/status
health
reports
```

Exact schemas belong in the production specification.

---

# 64. Async Processing

Heavy processing must not block FastAPI request handlers.

The intended lifecycle is approximately:

```text
QUEUED
  ↓
INGESTING
  ↓
PREPROCESSING
  ↓
CLASSIFYING
  ↓
SEGMENTING
  ↓
CHARACTERIZING
  ↓
FETCHING_ENVIRONMENT
  ↓
HINDCASTING
  ↓
FORECASTING
  ↓
FETCHING_AIS
  ↓
ATTRIBUTING
  ↓
COMPLETE
```

Failure/degraded states must also be supported.

Potential worker technologies include:

```text
Celery
RQ
Arq
```

The exact choice can be made during implementation.

---

# 65. Reproducibility

Each analysis should record enough information to reproduce or audit the result.

Important metadata includes:

```text
model version
model checksum
normalization statistics
classification threshold
satellite product ID
satellite acquisition time
preprocessing version
environment provider
environment dataset/model
simulation parameters
AIS provider
AIS retrieval time
scoring version
```

---

# 66. Database Direction

The intended production database is:

```text
PostgreSQL
+
PostGIS
```

Useful entities include:

```text
analyses
satellite products
spill geometries
environment datasets
simulations
AIS tracks
vessel candidates
vessel scores
jobs
provider events
```

Do not store huge raster/NetCDF blobs directly in relational tables.

---

# 67. Object Storage Direction

Large artifacts should use object storage/filesystem abstraction.

Examples:

```text
SAR products
segmentation masks
GeoJSON
NetCDF
simulation outputs
reports
```

The database stores metadata/references rather than unnecessarily storing large binary objects inline.

---

# 68. Secrets

Never hardcode:

```text
Copernicus credentials
Copernicus Marine credentials
AIS API keys
database passwords
JWT secrets
other provider credentials
```

Use:

```text
.env
secret manager
deployment environment
```

and provide:

```text
.env.example
```

without real credentials.

---

# 69. Old Prototype Artifacts

The previous development repository contained experimental code and an old U-Net checkpoint/demo:

```text
aquavision_unet.pth
```

That old implementation was considered hardcoded/prototype work.

Do NOT assume it should be used in the new production repository.

If the user wants to preserve it, it should be treated as an experimental reference/checkpoint, not as the canonical production model.

---

# 70. Current Training/Production Separation

There should be a clear separation between:

## Training environment

Primarily:

```text
Google Colab
Google Drive
90 GB dataset
GPU
```

and:

## Production inference environment

Primarily:

```text
AquaVision backend
trained model artifacts
satellite input
provider APIs
OpenDrift
AIS
database/object storage
```

The production server should not need the entire training dataset.

---

# 71. Current Priority

The project is currently moving from:

```text
prototype
```

to:

```text
deployment-stage demo
```

The immediate priority is not to build every possible feature at once.

The implementation should first establish a reliable vertical slice:

```text
Input Sentinel-1 scene
        ↓
ResNet gate
        ↓
U-Net
        ↓
spill geometry
        ↓
environmental forcing
        ↓
OpenDrift
        ↓
origin region
        ↓
AIS
        ↓
vessel ranking
```

Then add monitoring and production hardening.

---

# 72. Recommended Implementation Order

The production implementation should broadly follow:

```text
PHASE 1
Repository + backend foundation

PHASE 2
Production ML Stage 1

PHASE 3
Historical analysis API

PHASE 4
Environmental provider adapters

PHASE 5
OpenDrift hindcast/forecast

PHASE 6
Simulation API + frontend integration

PHASE 7
AIS integration

PHASE 8
Vessel attribution/ranking

PHASE 9
Monitoring / CDSE ingestion

PHASE 10
Security, testing, observability, deployment
```

Do not build monitoring before the historical pipeline is reliable.

---

# 73. What Is Already Validated vs Not Validated

## Validated experimentally

- Sentinel-1-style source TIFFs can be read with rasterio.
- Source images are 2-channel float32.
- Masks are binary uint8 matrices.
- 2048 × 2048 scenes can be tiled into 512 × 512 patches.
- Patch labels can be derived from mask oil fraction.
- Scene-level splitting works.
- ResNet18 can be adapted from 3 channels to 2 channels.
- A binary ResNet gate can learn a useful signal on the development subset.
- U-Net can produce promising segmentation results on the development setup.
- Lazy rasterio window loading works conceptually.

## NOT yet validated as production performance

- Full 90 GB ResNet performance.
- Final full-dataset U-Net performance.
- Final operating threshold.
- Generalization to unseen Sentinel-1 scenes outside the SIH dataset.
- Exact production preprocessing equivalence between training imagery and CDSE GRD products.
- End-to-end OpenDrift accuracy for a real incident.
- Accuracy of inferred spill origin.
- Accuracy of AIS attribution.
- Legal/forensic validity of vessel attribution.
- Continuous operational monitoring.

Claude Code must preserve this distinction.

---

# 74. Do Not Manufacture Results

Claude Code must never invent:

- accuracy
- F1
- IoU
- Dice
- vessel scores
- origin coordinates
- environmental conditions
- AIS records
- satellite detections
- API responses
- simulation outputs

If a provider is unavailable, return a clear degraded/error state.

If data is unavailable, say so.

Do not silently substitute fabricated values.

For UI demonstrations, clearly mark synthetic/demo fixtures as synthetic.

---

# 75. Demo vs Real Data

AquaVision may eventually need a polished demonstration environment.

If synthetic fixtures are used:

```text
DEMO / SYNTHETIC
```

must be distinguishable from:

```text
LIVE / REAL DATA
```

Never make simulated vessel tracks or simulated oil-spill trajectories appear to be real observations.

---

# 76. Expected Scientific Language

Prefer:

```text
Potential oil spill detected
```

over:

```text
Oil spill confirmed
```

Prefer:

```text
Vessel of interest
```

over:

```text
Guilty vessel
```

Prefer:

```text
Likely origin region
```

over:

```text
Exact origin
```

Prefer:

```text
Forecast probability / projected movement
```

over:

```text
Guaranteed movement
```

---

# 77. Current Conceptual Architecture

The final architecture should conceptually look like:

```text
                    ┌─────────────────────┐
                    │ Copernicus /       │
                    │ Sentinel-1         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Stage 1            │
                    │ Preprocess          │
                    │ + ResNet18          │
                    │ + U-Net             │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Spill              │
                    │ Characterization    │
                    └──────────┬──────────┘
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
       ┌───────────────────┐      ┌───────────────────┐
       │ Copernicus Marine │      │ Weather Provider  │
       │ Ocean Currents    │      │ Wind              │
       └─────────┬─────────┘      └─────────┬─────────┘
                 └────────────┬────────────┘
                              ▼
                    ┌─────────────────────┐
                    │ OpenDrift / OpenOil │
                    │ Hindcast + Forecast │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        Origin region/time             Forecast
                 │
                 ▼
          ┌───────────────┐
          │ AIS Provider  │
          └───────┬───────┘
                  ▼
          ┌───────────────┐
          │ AIS Processing│
          │ + Trajectory  │
          └───────┬───────┘
                  ▼
          ┌───────────────┐
          │ Vessel Scoring│
          └───────┬───────┘
                  ▼
          ┌───────────────┐
          │ AquaVision UI │
          └───────────────┘
```

---

# 78. Fresh Repository Rule

This repository should be considered empty/new unless files are actually present.

Claude Code must:

1. inspect the repository
2. inspect existing files
3. read this context
4. read `AquaVision_Claude_Code_Production_Spec.md`
5. compare current repository against the intended architecture
6. propose changes
7. implement incrementally

Do not assume previous repository files exist.

---

# 79. First Claude Code Instruction

The recommended first prompt after placing this file and the production specification in the repository is:

```text
Read these two documents completely:

1. @AQUAVISION_PROJECT_CONTEXT.md
2. @AquaVision_Claude_Code_Production_Spec.md

The first document describes work that has already been completed
in an earlier AquaVision repository.

The second document describes the production system we now want
to build.

This repository is intentionally a fresh production repository.

Do NOT assume that code from the previous repository exists here.
Do NOT recreate experimental code blindly.

Use the project context to understand:
- what has already been experimentally validated
- what decisions have already been made
- what remains unvalidated
- what must be rebuilt cleanly

First inspect the current repository.

Then give me:
1. a repository audit
2. a gap analysis against the production specification
3. the proposed implementation phases
4. the exact files/directories you intend to create or modify
5. any technical risks or missing information

Do NOT modify files yet.

Wait for approval before implementing.
```

---

# 80. Final Instruction to Claude Code

The goal is not merely to create a visually impressive demo.

The goal is to build a credible, modular, reproducible AquaVision prototype that can demonstrate the complete chain:

```text
Satellite observation
        ↓
Potential oil detection
        ↓
Segmentation
        ↓
Physical characterization
        ↓
Environmental reconstruction
        ↓
Hindcast
        ↓
Origin probability
        ↓
Forecast
        ↓
AIS correlation
        ↓
Vessel-of-interest ranking
        ↓
Response decision support
```

Every stage must use real inputs where available, expose uncertainty, preserve provenance, and fail honestly when data or providers are unavailable.

The production specification is the implementation authority.

This document is the historical/project-context authority.

When the two appear to conflict:

1. Preserve scientifically validated facts from this context.
2. Prefer the cleaner production architecture in the production specification.
3. Do not invent missing information.
4. Ask for clarification when a conflict materially affects implementation.
