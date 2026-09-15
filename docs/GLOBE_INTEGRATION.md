# Globe integration

## Archive inspected

`AquaVision_Globe.zip` was inspected from the source files, not screenshots. The supplied project is a React/Vite frontend with these relevant areas:

- `src/components/Globe/AquaGlobe.jsx`
- `src/components/Map/AquaMap.jsx`
- `src/components/Panels/HistoricPanel.jsx`
- `src/components/Panels/SystemStatus.jsx`
- `src/components/TopBar.jsx`
- `src/data/mockDats.js`
- `src/data/satellite.js`
- `src/services/api.js`
- `src/assets/hero.png`

The archive also contains its own `.git`, `node_modules`, and build/dependency material. None of those were migrated.

## Technology and behavior

The supplied globe uses `globe.gl`, Three.js, and `satellite.js`. `AquaGlobe.jsx` creates a globe instance, applies Earth imagery and topology textures, configures a non-auto-rotating orbit camera, propagates hard-coded Sentinel satellite TLEs, and renders satellite objects and a selected satellite footprint. The supplied 2D view uses Leaflet/react-leaflet and contains static spill polygons and origin points.

The supplied API service is empty. The supplied panels and data modules contain static/demo values, including satellite TLEs, vessel/event-style state, and map coordinates.

## Reused implementation

The current frontend now contains adapted `src/components/AquaGlobe.jsx` and `src/components/AquaMap.jsx` components in one application. The globe preserves the useful behavior:

- `globe.gl` Earth rendering
- Three-dimensional orbit/pan/zoom controls
- Earth imagery and topology textures
- Point labels and click selection
- Polygon rendering
- Animated dashed trajectory paths
- Responsive canvas resizing

The 2D map preserves the archive's Leaflet/react-leaflet approach with pan/zoom and popups. It renders the same backend-provided origin, forecast, vessel position, and vessel track data as the globe.

## Adaptations

Both migrated visualizations are prop-driven. They receive the existing FastAPI analysis result and render only layers available in that response:

- AIS candidate positions
- AIS candidate tracks
- probable origin polygons
- forecast trajectories

The static satellite TLE records, static spill polygon, static vessel names, and static map coordinates from the archive were not copied. The backend currently does not supply satellite position/orbit geometry or segmentation geometry, so those layers remain absent rather than fabricated.

The surrounding dashboard keeps the existing AquaVision frontend entry point and connects to:

- `GET /api/v1/readiness`
- `POST /api/v1/analysis`

No second React application was created.

## Data and state decisions

- No analysis: show `No active analysis` and an empty globe state.
- Loading: disable the run action and show `Running...`.
- Demo response: show `DEMO DATA`; values come from the backend response but remain explicitly demo-labelled.
- Missing live configuration: show `DEMO MODE` and readiness blockers.
- Vessel selection: show `Vessel of interest`; the UI does not claim causation.
- Missing stages: show unavailable messages rather than generated operational values.

The current backend analysis route still executes the demo engine. Production readiness therefore remains false until real Stage 1 orchestration and provider execution are connected.
