import os
import ee
import requests
from datetime import datetime, timedelta

def initialize_ee():
    """Initializes Earth Engine using your registered Cloud Project."""
    try:
        print("   -> [GEE] Initializing Earth Engine with project genial-union-472421-s5...")
        ee.Initialize(project='genial-union-472421-s5')
        print("   -> [GEE] Authentication & Initialization successful.")
    except Exception as e:
        print("   -> [GEE ERROR] Earth Engine failed to initialize.")
        raise e

def fetch_sentinel_image(lat: float, lon: float, target_date: str = None, output_dir: str = "data/temp"):
    """
    Queries Google Earth Engine for the latest Sentinel-1 GRD image over a coordinate.
    Downloads a 512x512 GeoTIFF and extracts the exact acquisition timestamp.
    """
    initialize_ee()
    os.makedirs(output_dir, exist_ok=True)

    # 2560m buffer to crop ~5km x 5km box (512x512 at 10m/px)
    point = ee.Geometry.Point([lon, lat])
    aoi = point.buffer(2560).bounds()

    print("   -> [GEE] Querying COPERNICUS/S1_GRD collection...")
    collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(aoi) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW'))

    # Time Filtering
    if target_date:
        dt = datetime.fromisoformat(target_date.replace("Z", ""))
        start_date = (dt - timedelta(days=2)).strftime('%Y-%m-%d')
        end_date = (dt + timedelta(days=2)).strftime('%Y-%m-%d')
        collection = collection.filterDate(start_date, end_date)
    else:
        # Check the last 30 days to guarantee a satellite pass over this location
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)
        collection = collection.filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    latest_image = collection.sort('system:time_start', False).first()

    print("   -> [GEE] Requesting image metadata from Google...")
    image_info = latest_image.getInfo()

    if not image_info:
        raise ValueError(f"No Sentinel-1 pass found near coordinates ({lat}, {lon}) in the selected timeframe.")

    timestamp_ms = image_info.get('properties', {}).get('system:time_start')
    acquisition_time = datetime.utcfromtimestamp(timestamp_ms / 1000.0).isoformat() + "Z"
    print(f"   -> [GEE] Found scene captured at: {acquisition_time}")

    image_export = latest_image.select(['VV', 'VH']).clip(aoi)

    print("   -> [GEE] Fetching download URL...")
    download_url = image_export.getDownloadURL({
        'scale': 10,
        'crs': 'EPSG:4326',
        'region': aoi,
        'format': 'GEO_TIFF'
    })

    print("   -> [GEE] Downloading GeoTIFF directly from Google...")
    response = requests.get(download_url, timeout=60)
    if response.status_code != 200:
        raise Exception(f"GEE download request failed with HTTP status {response.status_code}")

    extracted_path = os.path.join(output_dir, "latest_scene.tif")
    
    # Write the raw TIFF bytes directly to disk (No unzipping needed!)
    with open(extracted_path, 'wb') as f:
        f.write(response.content)

    print(f"   -> [GEE] Saved extracted scene to: {extracted_path}")

    return {
        "filepath": extracted_path,
        "acquisition_time": acquisition_time,
        "latitude": lat,
        "longitude": lon
    }