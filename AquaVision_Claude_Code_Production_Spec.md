# AquaVision --- Production Implementation Specification for Claude Code

## 0. Purpose

This document is the implementation contract for building the AquaVision
SIH 2026 PS 26143 prototype as a deployable, production-grade full-stack
system.

**Problem statement:** SIH 2026 PS 26143 --- "Leveraging satellite
imagery to determine Oil spills at sea along with AIS data correlations
to identify vessel responsible for the spill." The official problem
statement asks for an automated pipeline that detects and characterises
oil slicks, uses oceanographic and meteorological data for
backward/forward drift modelling, and analyses historic AIS traffic to
rank potential suspect vessels using proximity, trajectory and
behavioural anomalies. It also calls for a suitable visual interface.

**Core product flow:**

> **SEE → RECONSTRUCT → TRACE → ATTRIBUTE → RESPOND**

The implementation must be modular. ML training, satellite ingestion,
physical simulation, AIS correlation, API integrations, backend
orchestration, and UI must be independently testable.

This document is intentionally written for **Claude Code CLI**. Claude
Code should implement the repository, tests, configuration, migrations,
API contracts, frontend, workers, and deployment artifacts rather than
producing isolated demo snippets.

------------------------------------------------------------------------

# 1. Non-negotiable engineering rules

## 1.1 Do not fake scientific results

Never invent:

-   satellite acquisition times
-   vessel positions
-   weather observations
-   ocean currents
-   spill origins
-   AIS records
-   model confidence
-   simulation trajectories
-   API responses

If an external provider is unavailable, the system must return an
explicit degraded state such as:

``` json
{
  "status": "degraded",
  "provider": "copernicus_marine",
  "reason": "credentials_missing"
}
```

For development, deterministic fixtures may be used, but they must be
clearly marked as fixtures and must never be presented as live data.

## 1.2 Do not claim that SAR alone proves oil

Stage 1 produces a **potential oil-spill candidate**. SAR dark
signatures can have lookalikes.

The UI and API should therefore use terminology such as:

-   "Potential oil spill"
-   "Oil candidate"
-   "Detection confidence"
-   "Segmentation confidence"

rather than "confirmed oil".

## 1.3 Do not claim AIS proves legal responsibility

AIS correlation produces:

-   vessel of interest
-   correlation score
-   evidence factors
-   confidence

It must not state that a vessel is legally responsible unless
independent evidence establishes that.

Use:

-   "High-priority vessel of interest"
-   "High correlation"
-   "Strong spatio-temporal match"

instead of "confirmed polluter".

## 1.4 Do not hard-code API credentials

All credentials must come from environment variables or a secrets
manager.

Never commit:

-   API keys
-   passwords
-   Copernicus credentials
-   database credentials
-   JWT secrets

## 1.5 Keep providers behind adapters

Do not couple the application directly to one weather, marine-current,
AIS, or satellite provider.

Use interfaces such as:

``` text
SatelliteCatalogProvider
WeatherProvider
OceanCurrentProvider
AISProvider
```

This allows providers to be changed without rewriting the application.

------------------------------------------------------------------------

# 2. Target repository architecture

Create a clean repository:

``` text
aquavision/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── health.py
│   │   │   │   ├── analysis.py
│   │   │   │   ├── monitoring.py
│   │   │   │   ├── simulations.py
│   │   │   │   ├── ais.py
│   │   │   │   └── jobs.py
│   │   │   └── dependencies.py
│   │   ├── schemas/
│   │   ├── services/
│   │   │   ├── stage1/
│   │   │   ├── satellite/
│   │   │   ├── weather/
│   │   │   ├── ocean/
│   │   │   ├── drift/
│   │   │   ├── ais/
│   │   │   └── scoring/
│   │   ├── providers/
│   │   │   ├── cdse/
│   │   │   ├── copernicus_marine/
│   │   │   ├── weather/
│   │   │   └── ais/
│   │   ├── workers/
│   │   ├── db/
│   │   ├── storage/
│   │   └── utils/
│   ├── tests/
│   ├── migrations/
│   ├── requirements.txt
│   └── Dockerfile
│
├── ml/
│   ├── resnet/
│   │   ├── dataset.py
│   │   ├── index_dataset.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── threshold.py
│   │   └── inference.py
│   ├── unet/
│   │   ├── dataset.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── inference.py
│   ├── preprocessing/
│   └── artifacts/
│       ├── resnet18/
│       └── unet/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── api/
│   │   ├── map/
│   │   ├── state/
│   │   └── types/
│   ├── public/
│   ├── tests/
│   └── Dockerfile
│
├── infra/
│   ├── docker/
│   ├── nginx/
│   └── deployment/
│
├── data/
│   ├── fixtures/
│   └── schemas/
│
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── operations.md
│
├── .env.example
├── docker-compose.yml
├── README.md
└── Makefile
```

The exact frontend framework may remain the existing team choice, but
the backend contract must be framework-independent.

------------------------------------------------------------------------

# 3. Stage 1 --- Satellite oil-spill detection

## 3.1 Input

Primary input:

``` text
Sentinel-1 SAR
VV + VH
```

The current training dataset consists of 2048×2048 two-channel TIFF
scenes.

The model pipeline works on:

``` text
512 × 512
```

patches.

The production ingestion layer must convert incoming Sentinel-1 products
into the same representation used for model training.

**Do not feed arbitrary raw GRD values directly into the model.**

The preprocessing contract must explicitly record:

-   product ID
-   acquisition timestamp
-   orbit/pass information when available
-   CRS
-   polarization
-   calibration/preprocessing version
-   VV/VH representation
-   scaling/normalization metadata

------------------------------------------------------------------------

# 4. ResNet18 candidate gate

## 4.1 Purpose

ResNet18 is a binary gate:

``` text
VV + VH patch
       ↓
ResNet18
       ↓
P(OIL)
```

Classes:

``` text
0 = NOT OIL
1 = OIL
```

NOT OIL includes:

-   no-oil patches
-   lookalike patches

It is not a 3-class final classifier.

## 4.2 Input

``` text
2 channels
VV
VH

512 × 512
```

Modify the first ResNet18 convolution from 3 input channels to 2.

Initialize the new convolution from pretrained RGB weights by averaging
the original RGB input-channel weights and repeating that average for
VV/VH initialization.

Final layer:

``` text
Linear(512, 1)
```

Output is a logit; inference applies sigmoid.

## 4.3 Normalization

Do not use ImageNet normalization.

Compute:

``` text
VV mean
VV std
VH mean
VH std
```

from training scenes only.

Persist these values with the model artifact.

## 4.4 Patch labels

For Oil scenes:

``` text
oil_fraction =
oil_mask_pixels / total_patch_pixels
```

Initial training rule:

``` text
oil_fraction >= 0.01
→ label 1
```

otherwise:

``` text
label 0
```

This threshold must remain configurable.

## 4.5 Leakage prevention

Split by **scene**, never by individual patch.

Correct:

``` text
scenes
 ↓
train / validation
 ↓
patches
```

Incorrect:

``` text
all patches
 ↓
random train/validation split
```

Part III must remain untouched for final evaluation.

## 4.6 Training

Initial configuration:

``` text
Model: ResNet18
Input: 2 channels
Loss: BCEWithLogitsLoss with positive class weighting
Optimizer: AdamW
Initial LR: 1e-4
Weight decay: 1e-4
Scheduler: ReduceLROnPlateau or equivalent
```

Track:

-   train loss
-   validation loss
-   precision
-   recall
-   F1
-   PR-AUC
-   confusion matrix

PR-AUC is especially important because the positive class is sparse.

## 4.7 Threshold

Do not use 0.5 automatically.

The small development experiment produced approximately:

``` text
threshold 0.14 → precision 0.846, recall >= 0.70
threshold 0.11 → precision 0.800, recall >= 0.80
threshold 0.05 → precision 0.295, recall >= 0.90
```

These values are from a small development validation set and are not
final.

For production, select the threshold on the full validation set.

Because ResNet is a gate before U-Net:

> Prefer high recall over maximum precision.

The threshold must be stored in model metadata:

``` json
{
  "classifier_threshold": 0.12
}
```

Do not hard-code it in business logic.

------------------------------------------------------------------------

# 5. U-Net segmentation

## 5.1 Purpose

The U-Net receives candidate patches from ResNet and determines the
pixel-level spill region.

``` text
candidate patch
      ↓
U-Net
      ↓
pixel probability map
      ↓
binary segmentation mask
```

Input:

``` text
VV + VH
```

Output:

``` text
1-channel spill probability mask
```

## 5.2 Existing architecture

The earlier prototype used:

``` text
Input: 2 channels
Encoder: 2 → 32 → 64 → 128
Bottleneck: 128 → 256
Decoder with skip connections
Output: 1 channel
```

Loss:

``` text
BCE + Dice
```

Optimizer:

``` text
AdamW
LR = 1e-4
```

Earlier development results reached approximately:

``` text
Dice ≈ 0.85
IoU ≈ 0.75
```

These are development results, not final production metrics.

The full-data U-Net must be independently evaluated.

## 5.3 U-Net inference threshold

Do not assume the classifier threshold and segmentation threshold are
the same.

Store:

``` text
classifier_threshold
segmentation_threshold
```

separately.

## 5.4 Patch-to-scene reconstruction

Candidate patches may overlap or be adjacent.

The system must:

1.  Run ResNet.
2.  Select candidate patches.
3.  Run U-Net.
4.  Place each U-Net mask back into scene coordinates.
5.  Merge masks.
6.  Remove tiny connected components using configurable post-processing.
7.  Preserve the original geospatial transform.
8.  Convert the final raster mask into vector geometry.

Output:

``` text
scene mask
spill polygons
```

------------------------------------------------------------------------

# 6. Spill characterization

From the final segmentation polygon calculate:

``` text
area
perimeter
centroid
bounding box
length
width
orientation
compactness
shape descriptors
```

Where geospatial conversion is possible, report area in:

``` text
m² / km²
```

Do not calculate physical area using raw pixel counts alone.

Use the scene's georeferencing and an appropriate geodesic/projected
calculation.

Store both:

``` text
pixel geometry
geographic geometry
```

when useful.

------------------------------------------------------------------------

# 7. Spill age estimation

If enough temporal/physical information exists, estimate an age/time
window.

Do not pretend that image timestamp minus an arbitrary constant equals
spill age.

Preferred representation:

``` text
estimated_age:
    lower_bound
    upper_bound
    confidence
    method
```

Example:

``` json
{
  "age_hours": {
    "min": 8,
    "max": 20,
    "confidence": 0.62
  }
}
```

If the evidence is insufficient:

``` json
{
  "age_hours": null,
  "status": "insufficient_evidence"
}
```

------------------------------------------------------------------------

# 8. Stage 1 API contract

Create:

``` http
POST /api/v1/analysis
```

The request should support either:

### Historical/local product

``` json
{
  "mode": "historical",
  "product_id": "...",
  "asset_id": "..."
}
```

or a monitored product:

``` json
{
  "mode": "monitoring",
  "product_id": "..."
}
```

The response should create an analysis job.

For production, do not keep a long-running HTTP request open while the
ML/simulation pipeline executes.

Return:

``` http
202 Accepted
```

with:

``` json
{
  "analysis_id": "...",
  "status": "queued"
}
```

Then:

``` http
GET /api/v1/analysis/{analysis_id}
```

returns progress/state.

------------------------------------------------------------------------

# 9. Stage 2 --- Environmental data retrieval

After Stage 1 produces a spill polygon/centroid and acquisition
timestamp, retrieve environmental forcing data.

We need two major environmental fields:

``` text
1. Ocean surface currents
2. Wind
```

Optional:

``` text
waves
sea-surface temperature
water temperature
diffusivity
```

OpenDrift is designed to consume current, wind and other environmental
forcing data through readers and can read CF-compliant NetCDF/remote
datasets. It also supports backward simulation through negative time
steps.

------------------------------------------------------------------------

# 10. Ocean current provider

Primary production provider:

## Copernicus Marine

Use the current Copernicus Marine Toolbox/API rather than building
against deprecated legacy services.

The current Copernicus Marine Toolbox provides programmatic catalogue
access, subsetting and original-file retrieval through CLI and Python
API. It supports efficient spatial/temporal subsetting, including
cloud-optimized data access.

Implement:

``` text
CopernicusMarineCurrentProvider
```

Interface:

``` python
class OceanCurrentProvider(Protocol):

    async def get_surface_currents(
        self,
        bbox: BBox,
        start_time: datetime,
        end_time: datetime,
    ) -> EnvironmentalDataset:
        ...
```

The provider should return normalized data that OpenDrift can consume.

Required variables should include surface current components equivalent
to:

``` text
eastward current
northward current
```

with:

``` text
time
latitude
longitude
```

and units normalized to:

``` text
m/s
```

Do not assume variable names from a provider. Map provider-specific
variables to a canonical internal schema.

------------------------------------------------------------------------

# 11. Weather/wind provider

Implement a provider abstraction:

``` text
WeatherProvider
```

The first implementation may use Open-Meteo for operational/demo weather
retrieval.

Open-Meteo provides hourly wind speed and wind direction at 10 m and
also provides historical/historical-forecast endpoints, making it useful
for both forecast and historical environmental context.

However, Open-Meteo should be treated as a configurable provider, not
permanently baked into the simulation engine.

Canonical output:

``` text
time
latitude
longitude
wind_u
wind_v
wind_speed
wind_direction
```

Convert wind direction/speed into vector components:

``` text
u = eastward component
v = northward component
```

in:

``` text
m/s
```

For a stricter operational deployment, allow an ECMWF/ERA5-compatible
NetCDF provider or another authoritative meteorological source to be
configured without changing OpenDrift orchestration.

------------------------------------------------------------------------

# 12. Environmental data provenance

Every simulation must store:

``` json
{
  "ocean_provider": "copernicus_marine",
  "ocean_dataset": "...",
  "ocean_dataset_version": "...",
  "weather_provider": "...",
  "weather_dataset": "...",
  "retrieved_at": "...",
  "time_range": {
    "start": "...",
    "end": "..."
  },
  "bbox": [...]
}
```

This is critical for reproducibility.

------------------------------------------------------------------------

# 13. Environmental data caching

Do not repeatedly download the same forcing data.

Cache by:

``` text
provider
dataset
bbox
time range
resolution
variables
```

Use content-addressed or deterministic cache keys.

Example:

``` text
environment/
    copernicus/
        {dataset_hash}.nc
    weather/
        {dataset_hash}.nc
```

Store metadata in the database.

------------------------------------------------------------------------

# 14. Stage 2 --- OpenDrift

Use:

``` python
from opendrift.models.openoil import OpenOil
```

for oil-specific simulations.

OpenDrift's OpenOil model is specifically designed for oil drift and
includes oil-specific properties and weathering processes. OpenDrift
readers provide environmental forcing such as current, wind and
temperature.

The simulation service must be isolated from the API process.

------------------------------------------------------------------------

# 15. Converting segmentation into OpenDrift seeds

Do not seed particles from a single arbitrary centroid only.

The segmented spill polygon is the observed state.

Generate an ensemble of seed points within/around the detected spill
polygon.

Possible strategy:

``` text
spill polygon
    ↓
sample N particles
    ↓
preserve spatial distribution
```

Example:

``` text
N = 5,000
```

with the exact count configurable.

Each particle represents uncertainty rather than an actual measured oil
droplet.

------------------------------------------------------------------------

# 16. Hindcast

Purpose:

> Estimate where the observed slick may have originated and during what
> time window.

Inputs:

``` text
observed spill geometry
observation timestamp
ocean currents
wind
OpenOil parameters
```

Concept:

``` text
Observed spill
      ↓
Seed particle ensemble
      ↓
Backward simulation
      ↓
particle cloud at earlier times
      ↓
origin probability density
```

OpenDrift supports backward simulation by running with a negative time
step.

Do not describe the output as an exact origin unless the evidence truly
supports that precision.

------------------------------------------------------------------------

# 17. Hindcast origin calculation

At each backward timestep calculate the particle distribution.

For example:

``` text
t - 1h
t - 2h
t - 3h
...
t - Nh
```

At each time:

1.  Collect valid particle positions.
2.  Generate a spatial density estimate.
3.  Calculate concentration regions.
4.  Identify the highest-density connected region.
5.  Calculate a weighted centroid.
6.  Store uncertainty radius/ellipse.
7.  Compare the reconstructed cloud against the observed spill geometry.

Output:

``` json
{
  "origin": {
    "latitude": 12.345,
    "longitude": 78.901,
    "time_start": "...",
    "time_end": "...",
    "confidence": 0.73,
    "uncertainty_km": 8.4
  }
}
```

This "origin point" is a **representative point inside an origin
probability region**, not a claim that the spill began at an exact
coordinate.

The UI should display the uncertainty region.

------------------------------------------------------------------------

# 18. Hindcast confidence

Calculate a transparent confidence score from measurable quantities.

Possible factors:

``` text
backward particle concentration
temporal stability of origin region
agreement between nearby timesteps
environmental data coverage
number of valid particles
simulation duration
```

Do not invent a statistical confidence interval without defining the
method.

Store:

``` text
origin_confidence
origin_uncertainty_km
origin_time_window
```

separately.

------------------------------------------------------------------------

# 19. Forecast

Purpose:

> Show responders where the spill may move next.

Start from the observed spill geometry or the current inferred particle
state.

Run:

``` text
current time
   ↓
+6 h
+12 h
+24 h
+48 h
+72 h
```

Exact horizons should be configurable.

Output:

``` text
particle trajectories
forecast density
forecast polygons
```

The UI should show an animated time slider.

------------------------------------------------------------------------

# 20. Forecast uncertainty

Do not display a single deterministic line as "the future path".

Display:

``` text
ensemble particle cloud
+
probability/density region
```

Optionally render:

``` text
50% region
75% region
90% region
```

if the statistical method is implemented correctly.

------------------------------------------------------------------------

# 21. OpenDrift output format

Persist the raw simulation output as CF-compliant NetCDF where
appropriate; OpenDrift supports NetCDF output.

Also produce an application-friendly JSON/GeoJSON representation:

``` text
simulation/
    raw.nc
    particles.geojson
    density.geojson
    summary.json
```

Do not force the browser to download and parse large NetCDF files.

------------------------------------------------------------------------

# 22. Backend simulation API

Create:

``` http
POST /api/v1/simulations
```

Request:

``` json
{
  "analysis_id": "...",
  "mode": "hindcast"
}
```

or:

``` json
{
  "analysis_id": "...",
  "mode": "forecast"
}
```

Return:

``` http
202 Accepted
```

with:

``` json
{
  "simulation_id": "...",
  "status": "queued"
}
```

Then:

``` http
GET /api/v1/simulations/{simulation_id}
```

and:

``` http
GET /api/v1/simulations/{simulation_id}/layers
```

------------------------------------------------------------------------

# 23. UI --- overall design

The UI should feel like an operational intelligence dashboard, not a
generic AI demo.

Primary navigation:

``` text
Dashboard
Historical Analysis
Monitoring
Simulation
Vessel Attribution
Reports
System Status
```

Main workflow:

``` text
Satellite
   ↓
Detection
   ↓
Segmentation
   ↓
Drift Reconstruction
   ↓
Vessel Attribution
   ↓
Response
```

------------------------------------------------------------------------

# 24. UI --- Historical Mode

Historical mode should allow:

``` text
Select existing Sentinel-1 scene
        ↓
Analyze
```

Display:

### Left/main map

-   Sentinel-1 footprint
-   spill polygon
-   centroid
-   origin probability region
-   hindcast trajectories
-   forecast region
-   vessels

### Right analysis panel

``` text
POTENTIAL OIL SPILL

Detection confidence
Segmentation confidence
Estimated area
Acquisition time
Location
```

Then:

``` text
ORIGIN RECONSTRUCTION

Likely origin region
Estimated time window
Origin confidence
Uncertainty
```

Then:

``` text
VESSEL ATTRIBUTION

#1 Vessel
Score
Distance
Time difference
Trajectory correlation
Behavior anomaly

#2 Vessel
...
```

------------------------------------------------------------------------

# 25. UI --- Monitoring Mode

Monitoring mode:

``` text
AOI selection
      ↓
Search Sentinel-1
      ↓
New products
      ↓
Analyze
```

The UI should show:

``` text
Monitoring status
Last catalogue check
Latest acquisition
Processing state
```

Do not call it "live satellite video".

Use:

> Near-real-time Sentinel-1 monitoring

------------------------------------------------------------------------

# 26. CDSE satellite ingestion

Use Copernicus Data Space Ecosystem catalogue APIs.

CDSE currently exposes OData and STAC catalogue interfaces. OData
supports product filtering by collection, acquisition time and other
metadata; subscriptions can be used to receive notifications about newly
added relevant products. The current STAC endpoint is:

``` text
https://stac.dataspace.copernicus.eu/v1/
```

The legacy CDSE STAC endpoint has been deprecated since November 2025.

Implement:

``` text
CDSECatalogProvider
```

with:

``` python
search_products(
    bbox,
    start_time,
    end_time,
    platform="SENTINEL-1",
    product_type="GRD"
)
```

Return normalized:

``` text
product_id
product_name
acquisition_time
geometry
orbit
polarizations
download/access metadata
```

------------------------------------------------------------------------

# 27. CDSE monitoring worker

Do not make the browser poll CDSE directly.

Backend worker:

``` text
scheduler
    ↓
CDSE query/subscription
    ↓
new product detected
    ↓
deduplicate by product ID
    ↓
create analysis job
```

Store:

``` text
product_id
first_seen_at
processed_at
processing_status
analysis_id
```

A product must never be processed twice.

------------------------------------------------------------------------

# 28. AIS Stage 3

After hindcast:

``` text
origin probability region
+
origin time window
```

query AIS.

The official problem statement specifically allows sample AIS from
MarineCadastre and says real AIS may be used if available, otherwise
synthetic data may demonstrate the algorithm.

Implement both:

``` text
HistoricalAISProvider
LiveAISProvider
```

behind one interface.

------------------------------------------------------------------------

# 29. Historical AIS

Use MarineCadastre-compatible historical AIS data where appropriate.

The provider must support:

``` python
query_tracks(
    bbox,
    start_time,
    end_time
)
```

Normalize vessel records into:

``` text
mmsi
imo
ship_name
timestamp
latitude
longitude
sog
cog
heading
nav_status
```

Do not assume every field exists for every AIS message.

------------------------------------------------------------------------

# 30. Live AIS

For a live demonstration, a provider such as AISStream can be integrated
behind:

``` text
AISStreamProvider
```

AISStream currently exposes a WebSocket stream at:

``` text
wss://stream.aisstream.io/v0/stream
```

and supports bounding-box and message-type subscriptions. API keys must
remain server-side.

Never put an AIS API key in the frontend.

------------------------------------------------------------------------

# 31. AIS query strategy

Do not download every vessel in an ocean.

First determine an ROI:

``` text
origin probability region
+
spatial buffer
```

and:

``` text
origin time window
+
temporal buffer
```

Example configurable defaults:

``` text
spatial buffer: 25–100 km
temporal buffer: ±24 h
```

The actual values should be configuration-driven and justified by the
simulation uncertainty.

------------------------------------------------------------------------

# 32. AIS trajectory reconstruction

AIS messages are irregular.

For each vessel:

1.  Sort by timestamp.
2.  Remove impossible coordinates.
3.  Remove impossible speeds/jumps.
4.  Deduplicate messages.
5.  Interpolate only where scientifically justified.
6.  Build a time-ordered trajectory.
7.  Calculate derived metrics.

Derived metrics:

``` text
distance to origin
time difference to estimated release
speed
heading
heading change
speed change
stops
loitering
route deviation
turn rate
```

Do not interpolate across very long gaps without marking the
uncertainty.

------------------------------------------------------------------------

# 33. Vessel candidate filtering

First filter vessels using hard spatial/temporal criteria:

``` text
vessel was inside / near origin region
AND
vessel was present during origin time window
```

Then calculate soft evidence.

This dramatically reduces the candidate set.

------------------------------------------------------------------------

# 34. Vessel scoring

Implement a transparent weighted score.

Initial conceptual score:

``` text
Suspicion Score =
    proximity_score
  + temporal_score
  + trajectory_score
  + behavioral_score
```

All components normalized to:

``` text
0–100
```

Example configurable weights:

``` text
proximity       35%
temporal        30%
trajectory      20%
behavior        15%
```

Do not hard-code these as scientifically validated weights. Make them
configurable and document them.

------------------------------------------------------------------------

# 35. Proximity score

Measure vessel distance from the origin probability region.

Example concept:

``` text
very close → high score
far away   → low score
```

Use geodesic distance.

Do not use raw lat/lon Euclidean distance.

------------------------------------------------------------------------

# 36. Temporal score

Measure:

``` text
abs(vessel_time - inferred_origin_time)
```

relative to the origin uncertainty window.

Closer temporal match:

``` text
higher score
```

Outside the credible window:

``` text
near zero
```

------------------------------------------------------------------------

# 37. Trajectory score

Measure whether the vessel trajectory is consistent with the inferred
origin.

Examples:

-   vessel path intersects origin region
-   vessel approaches origin region
-   vessel leaves origin region
-   vessel's heading is compatible with movement
-   vessel trajectory overlaps reconstructed origin corridor

Do not claim causality.

------------------------------------------------------------------------

# 38. Behavioral anomaly score

Calculate only observable anomalies.

Examples:

``` text
unexpected stop
speed drop
sharp course change
unusual loitering
route deviation
```

Compare against the vessel's recent local trajectory rather than
assuming a global "normal".

Avoid simplistic rules such as:

> "Stopped = guilty."

A stop is merely an anomaly candidate.

------------------------------------------------------------------------

# 39. Evidence breakdown

Every vessel score must be explainable.

Example:

``` json
{
  "mmsi": "123456789",
  "score": 86,
  "rank": 1,
  "evidence": {
    "proximity": 92,
    "temporal": 88,
    "trajectory": 81,
    "behavior": 76
  }
}
```

The UI should expose these components.

This is much stronger than displaying:

``` text
AI says vessel guilty: 86%
```

Do not do that.

------------------------------------------------------------------------

# 40. Vessel ranking API

Create:

``` http
GET /api/v1/attribution/{analysis_id}
```

Response:

``` json
{
  "status": "complete",
  "origin": {...},
  "vessels": [
    {
      "rank": 1,
      "mmsi": "...",
      "imo": "...",
      "name": "...",
      "score": 86,
      "classification": "high_correlation",
      "evidence": {...},
      "trajectory": {...}
    }
  ]
}
```

------------------------------------------------------------------------

# 41. Map visualization

Use a production-grade WebGL map library suitable for large geospatial
layers.

Required layers:

``` text
Base map
Sentinel-1 footprint
SAR scene
Spill segmentation
Spill polygon
Origin probability region
Hindcast particles
Forecast density
AIS vessel tracks
Vessel markers
Selected vessel trajectory
```

Controls:

``` text
Layers
Opacity
Time slider
Playback
Legend
Reset view
```

------------------------------------------------------------------------

# 42. Simulation playback

The UI should not simply display one static OpenDrift screenshot.

Create an interactive simulation timeline:

``` text
<  ─────●──────────────  >
   -24h    0h      +72h
```

Modes:

``` text
Hindcast
Forecast
```

Hindcast:

``` text
observed spill
      ↓
particles travel backward
      ↓
origin probability region
```

Forecast:

``` text
current spill
      ↓
particles travel forward
      ↓
predicted affected region
```

Animation should be driven by timestamped GeoJSON/vector data returned
by the backend.

------------------------------------------------------------------------

# 43. \[FRONTEND NOTE\]

Frontend must never directly call:

-   Copernicus credentials
-   AIS API keys
-   Copernicus Marine credentials
-   OpenDrift
-   model files
-   database

Frontend communicates only with AquaVision backend APIs.

------------------------------------------------------------------------

# 44. Analysis state machine

Create an explicit job state machine:

``` text
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

Error state:

``` text
FAILED
```

Optional degraded state:

``` text
DEGRADED
```

Every state should expose:

``` text
started_at
completed_at
error
progress
```

------------------------------------------------------------------------

# 45. Background workers

Heavy tasks must not execute inside FastAPI request handlers.

Use a job queue.

Possible stack:

``` text
FastAPI
PostgreSQL
Redis
Celery/RQ/Arq
```

or another reliable Python task system.

Tasks:

``` text
satellite_download
stage1_inference
environment_fetch
hindcast
forecast
ais_fetch
ais_processing
vessel_scoring
```

------------------------------------------------------------------------

# 46. Database

Use PostgreSQL.

Recommended tables:

``` text
users
analyses
satellite_products
spill_detections
spill_geometries
environment_datasets
simulations
simulation_outputs
ais_tracks
vessel_candidates
vessel_scores
job_runs
provider_events
```

Use PostGIS if geospatial queries are required.

Store geometry in geographic coordinates with an explicit SRID.

------------------------------------------------------------------------

# 47. Storage

Large files should not live inside PostgreSQL.

Use object storage:

``` text
S3-compatible storage
```

or the selected deployment provider's blob storage.

Store:

``` text
SAR assets
segmentation masks
GeoJSON
NetCDF
simulation outputs
generated reports
```

Database stores:

``` text
object key
checksum
metadata
content type
```

------------------------------------------------------------------------

# 48. Reproducibility

Every analysis must record:

``` text
model version
model checksum
preprocessing version
classifier threshold
segmentation threshold
satellite product ID
weather provider/dataset
ocean provider/dataset
simulation parameters
AIS provider
scoring version
```

This lets the same analysis be reproduced later.

------------------------------------------------------------------------

# 49. Model artifact metadata

Never deploy only:

``` text
resnet18.pth
```

Deploy:

``` text
resnet18/
    model.pth
    metadata.json
```

Metadata:

``` json
{
  "model": "resnet18",
  "input_channels": 2,
  "input_size": [512, 512],
  "channels": ["VV", "VH"],
  "normalization": {
    "vv_mean": 0.0,
    "vv_std": 1.0,
    "vh_mean": 0.0,
    "vh_std": 1.0
  },
  "classifier_threshold": 0.12,
  "training_dataset_version": "...",
  "git_commit": "...",
  "created_at": "..."
}
```

Use actual values once final training is complete.

------------------------------------------------------------------------

# 50. API versioning

All backend routes:

``` text
/api/v1/...
```

Never expose unversioned production APIs.

------------------------------------------------------------------------

# 51. Security

Implement:

-   authentication
-   authorization
-   request validation
-   rate limiting
-   CORS restrictions
-   secure headers
-   secrets from environment
-   structured audit logging
-   upload validation
-   maximum upload sizes
-   path traversal protection
-   SSRF protection for remote URLs
-   API timeout limits

Do not allow arbitrary user-provided URLs to be fetched by the server
without validation.

------------------------------------------------------------------------

# 52. Observability

Use structured JSON logs.

Every request/job should have:

``` text
request_id
analysis_id
simulation_id
job_id
provider
duration_ms
status
```

Metrics:

``` text
analysis duration
ML inference duration
provider latency
OpenDrift runtime
AIS query duration
failed jobs
API errors
queue length
```

------------------------------------------------------------------------

# 53. External API resilience

Every external provider call must have:

``` text
timeout
retry with exponential backoff
retry limit
HTTP status handling
schema validation
circuit-breaker/degraded behavior
```

Never retry indefinitely.

Cache successful responses where appropriate.

------------------------------------------------------------------------

# 54. Testing strategy

## Unit tests

Test:

-   patch extraction
-   normalization
-   classifier threshold
-   segmentation post-processing
-   geometry calculations
-   geodesic distance
-   origin scoring
-   AIS filtering
-   vessel scoring

## Integration tests

Test:

``` text
analysis API
 ↓
stage1
 ↓
environment
 ↓
simulation
 ↓
AIS
 ↓
attribution
```

using fixtures.

## Provider tests

Use recorded/mock responses.

Do not hit production APIs in every CI run.

## End-to-end test

Maintain one small deterministic fixture:

``` text
fixture Sentinel-1 scene
fixture mask
fixture environmental data
fixture AIS
```

and verify that the complete pipeline produces a valid result.

------------------------------------------------------------------------

# 55. Production deployment

Use Docker.

Services:

``` text
frontend
backend
worker
redis
postgres/postgis
reverse-proxy
```

Optional:

``` text
object storage
monitoring
```

Production deployment should support:

``` text
health checks
readiness checks
graceful shutdown
persistent storage
database migrations
rolling restart
```

------------------------------------------------------------------------

# 56. Health endpoints

Create:

``` http
GET /health/live
GET /health/ready
GET /health/providers
```

Example:

``` json
{
  "status": "healthy",
  "database": "ok",
  "redis": "ok",
  "model": "loaded",
  "copernicus": "configured",
  "ais": "configured"
}
```

Do not expose secrets.

------------------------------------------------------------------------

# 57. Report generation

The final analysis should be exportable.

Generate:

``` text
PDF report
JSON
GeoJSON
```

Report sections:

``` text
1. Spill detection
2. Segmentation
3. Physical characteristics
4. Origin reconstruction
5. Hindcast
6. Forecast
7. AIS analysis
8. Vessel ranking
9. Evidence breakdown
10. Data provenance
11. Limitations
```

------------------------------------------------------------------------

# 58. Final user workflow

The production UI should support:

``` text
USER
 │
 ├── Historical Mode
 │       ↓
 │   Select Sentinel-1
 │       ↓
 │   Analyze
 │
 └── Monitoring Mode
         ↓
     Select AOI
         ↓
     Discover new Sentinel-1
         ↓
     Analyze
         │
         ▼
     STAGE 1
         │
         ├── ResNet18
         ├── U-Net
         └── Geometry
         │
         ▼
     STAGE 2
         │
         ├── Weather
         ├── Ocean currents
         ├── Hindcast
         └── Forecast
         │
         ▼
     STAGE 3
         │
         ├── AIS retrieval
         ├── Trajectory reconstruction
         ├── Behavioral analysis
         └── Vessel ranking
         │
         ▼
     RESPONSE DASHBOARD
```

------------------------------------------------------------------------

# 59. Final dashboard result

The main result should communicate:

## Potential Spill

``` text
Detected
Confidence
Area
Coordinates
Acquisition time
```

## Origin

``` text
Likely origin region
Time window
Origin confidence
Uncertainty
```

## Movement

``` text
Hindcast
Forecast
Affected area
```

## Vessels of Interest

``` text
Rank
Vessel
Score
Distance
Time correlation
Trajectory correlation
Behavior anomaly
```

## Evidence

Each score must be expandable.

------------------------------------------------------------------------

# 60. What Claude Code should implement first

Do not attempt everything simultaneously.

Implement in this exact order.

## Phase 1 --- Repository foundation

-   create monorepo
-   backend
-   frontend
-   Docker
-   environment config
-   PostgreSQL/PostGIS
-   Redis
-   migrations
-   health checks
-   logging

## Phase 2 --- ML Stage 1

-   full ResNet training pipeline
-   full U-Net training pipeline
-   artifact metadata
-   inference service
-   threshold configuration
-   segmentation reconstruction
-   geometry extraction

## Phase 3 --- Historical analysis API

Get this complete first:

``` text
uploaded/selected Sentinel-1
 ↓
ResNet
 ↓
U-Net
 ↓
geometry
 ↓
API result
 ↓
UI map
```

## Phase 4 --- Environmental providers

Implement:

``` text
Copernicus Marine
Weather provider
```

with adapters, caching and provenance.

## Phase 5 --- OpenDrift

Implement:

``` text
seed generation
hindcast
origin density
origin point/region
forecast
GeoJSON output
NetCDF output
```

## Phase 6 --- UI simulation

Implement:

``` text
map
hindcast playback
forecast playback
origin uncertainty
environmental metadata
```

## Phase 7 --- AIS

Implement:

``` text
historical AIS
live AIS adapter
trajectory reconstruction
candidate filtering
scoring
ranking
```

## Phase 8 --- Final dashboard

Integrate:

``` text
satellite
spill
origin
simulation
AIS
ranking
report
```

## Phase 9 --- Monitoring

Implement:

``` text
CDSE catalogue search
new-product detection
deduplication
background processing
notification/status
```

## Phase 10 --- Production hardening

-   tests
-   retries
-   authentication
-   logging
-   monitoring
-   deployment
-   backups
-   error states
-   provider health
-   documentation

------------------------------------------------------------------------

# 61. Definition of done

AquaVision is considered deployment-ready only when this complete path
works:

``` text
Sentinel-1 product
       ↓
SAR preprocessing
       ↓
ResNet18 candidate detection
       ↓
U-Net segmentation
       ↓
Spill geometry
       ↓
Environmental data retrieval
       ↓
OpenDrift hindcast
       ↓
Origin probability region/time
       ↓
OpenDrift forecast
       ↓
AIS retrieval
       ↓
AIS trajectory reconstruction
       ↓
Candidate filtering
       ↓
Evidence scoring
       ↓
Ranked vessels of interest
       ↓
Production map UI
       ↓
Exportable report
```

And every stage must have:

``` text
real data
real provenance
error handling
logging
tests
configuration
```

------------------------------------------------------------------------

# 62. Important scientific/product language

Use:

> Potential oil spill

not:

> Confirmed oil spill

Use:

> Likely origin region

not:

> Exact origin point

unless the evidence supports exact localization.

Use:

> Vessel of interest / high correlation

not:

> Guilty vessel

Use:

> Forecast trajectory / probability region

not:

> Guaranteed future path

Use:

> Near-real-time monitoring

not:

> Live satellite video

------------------------------------------------------------------------

# 63. Official/current technical references

Claude Code should consult current official documentation before
implementing provider-specific calls because APIs change.

### Copernicus Data Space

OData:

https://documentation.dataspace.copernicus.eu/APIs/OData.html

STAC:

https://documentation.dataspace.copernicus.eu/APIs/STAC.html

On-Demand Production:

https://documentation.dataspace.copernicus.eu/APIs/On-Demand%20Production%20API.html

### Copernicus Marine

Current Toolbox:

https://help.marine.copernicus.eu/en/articles/7949409-copernicus-marine-toolbox-introduction

Python/API:

https://help.marine.copernicus.eu/en/articles/8286798-copernicus-marine-toolbox-api-explore-the-catalogue-and-metadata

Subset:

https://help.marine.copernicus.eu/en/articles/8283072-copernicus-marine-toolbox-api-subset

### OpenDrift

https://opendrift.github.io/

Tutorial:

https://opendrift.github.io/tutorial.html

OpenOil:

https://opendrift.github.io/autoapi/opendrift/models/openoil/openoil/index.html

### Open-Meteo

https://open-meteo.com/en/docs

Historical weather:

https://open-meteo.com/en/docs/historical-weather-api

### AISStream

https://aisstream.io/documentation

### MarineCadastre

Use the AIS source specified by the SIH problem statement for historical
AIS where applicable.

------------------------------------------------------------------------

# 64. Final instruction to Claude Code

Before writing implementation code:

1.  Inspect the existing repository.
2.  Inspect existing backend/frontend code.
3.  Do not delete working functionality without identifying it.
4.  Create a migration plan if legacy code exists.
5.  Build the new architecture incrementally.
6.  Keep provider integrations behind interfaces.
7.  Write tests alongside implementation.
8.  Use real APIs only where credentials/configuration are available.
9.  Use deterministic fixtures for tests.
10. Never fabricate external data.
11. Never expose secrets.
12. Never silently swallow provider/model errors.
13. Preserve provenance for every analysis.
14. Keep long-running operations asynchronous.
15. Return stable versioned API contracts.
16. Make thresholds and scientific parameters configurable.
17. Make all scoring weights configurable.
18. Keep the UI driven by backend data contracts rather than directly
    coupling it to ML/provider internals.
19. Prefer scientifically interpretable evidence over a single opaque
    "AI score".
20. At every stage, make the system demonstrable even if a later
    external provider is temporarily unavailable.

The final implementation should behave like an **operational
decision-support prototype**, not merely a collection of ML notebooks.
