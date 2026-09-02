# 1. Create Backend Folders
mkdir -p aquavision-backend/app/api, aquavision-backend/app/core, aquavision-backend/app/ml/sar, `
         aquavision-backend/app/ml/metocean, aquavision-backend/app/ml/ais, `
         aquavision-backend/app/ml/attribution, aquavision-backend/app/schemas, `
         aquavision-backend/data, aquavision-backend/models, aquavision-backend/static/overlays

# 2. Create Backend Blank Files
New-Item -ItemType File -Force -Path `
  "aquavision-backend/app/__init__.py", "aquavision-backend/app/api/__init__.py", "aquavision-backend/app/api/routes.py", `
  "aquavision-backend/app/core/__init__.py", "aquavision-backend/app/core/config.py", `
  "aquavision-backend/app/ml/__init__.py", `
  "aquavision-backend/app/ml/sar/__init__.py", "aquavision-backend/app/ml/sar/sentinel_service.py", "aquavision-backend/app/ml/sar/processor.py", `
  "aquavision-backend/app/ml/metocean/__init__.py", "aquavision-backend/app/ml/metocean/hindcast_engine.py", `
  "aquavision-backend/app/ml/ais/__init__.py", "aquavision-backend/app/ml/ais/ais_anomaly.py", `
  "aquavision-backend/app/ml/attribution/__init__.py", "aquavision-backend/app/ml/attribution/scorer.py", `
  "aquavision-backend/app/schemas/__init__.py", "aquavision-backend/app/schemas/payload.py", `
  "aquavision-backend/main.py", "aquavision-backend/requirements.txt"

# 3. Create Frontend Scaffold and Component Files
npm create vite@latest aquavision-frontend -- --template react -y
mkdir -p aquavision-frontend/src/api, aquavision-frontend/src/components
New-Item -ItemType File -Force -Path `
  "aquavision-frontend/src/api/client.js", `
  "aquavision-frontend/src/components/MapViewer.jsx", `
  "aquavision-frontend/src/components/SpillSummary.jsx", `
  "aquavision-frontend/src/components/CulpritLeaderboard.jsx", `
  "aquavision-frontend/src/components/MetoceanSimulator.jsx", `
  "aquavision-frontend/src/components/AnalyticsModal.jsx"


cd aquavision-frontend
npm install
npm install leaflet react-leaflet axios lucide-react
npm install -D tailwindcss @tailwindcss/postcss postcss autoprefixer

cd C:\Users\geeth\Downloads\AquaVision\aquavision-backend
python -m venv .origin
.\.origin\Scripts\activate
pip install fastapi uvicorn rasterio shapely opencv-python torch torchvision scikit-learn pandas netCDF4 Pillow earthengine-api
earthengine authenticate