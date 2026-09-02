import os
import ee
import requests
import zipfile
import io
from datetime import datetime, timedelta

def initialize_ee():
    """Initializes Earth Engine, catching authentication errors."""
    try:
        ee.Initialize()
    except Exception as e:
        print("Earth Engine not authenticated. Please run 'earthengine authenticate' in your terminal.")
        raise e

def fetch_sentinel_image(lat: float, lon: float, target_date: str = None, output_dir: str = "data/temp"):
    """
    Queries Google Earth Engine for the latest Sentinel-1 GRD image over a specific coordinate.
    Downloads a 512x512 (approx 5km x 5km) GeoTIFF and extracts the exact acquisition timestamp.
    """
    initialize_ee()
    os.makedirs(output_dir, exist_ok=True)

    # Define the point and a 2500m buffer to get a ~5km x 5km bounding box (512x512 pixels at 10m/px)
    point = ee.Geometry.Point([lon, lat])
    aoi = point.buffer(2560).bounds()

    # Base Sentinel-1 GRD Collection
    collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(aoi) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .filter(ee.Filter.eq('instrumentMode', 'IW'))

    # Time Filtering
    if target_date:
        # If user provides a date, look at that specific 48-hour window
        dt = datetime.fromisoformat(target_date.replace("Z", ""))
        start_date = (dt - timedelta(days=1)).strftime('%Y-%m-%d')
        end_date = (dt + timedelta(days=1)).strftime('%Y-%m-%d')
        collection = collection.filterDate(start_date, end_date)
    else:
        # If no date, look back up to 14 days for the latest available pass
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=14)
        collection = collection.filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    # Sort descending to get the newest image
    latest_image = collection.sort('system:time_start', False).first()

    # Check if an image was found
    if not latest_image.getInfo():
        raise ValueError(f"No Sentinel-1 imagery found for coordinates {lat}, {lon} in the given timeframe.")

    # 1. Extract Exact Temporal Metadata
    timestamp_ms = latest_image.get('system:time_start').getInfo()
    acquisition_time = datetime.utcfromtimestamp(timestamp_ms / 1000.0).isoformat() + "Z"

    # 2. Extract specific bands and clip to our Area of Interest
    image_export = latest_image.select(['VV', 'VH']).clip(aoi)

    # 3. Generate Download URL for the GeoTIFF
    print(f"Downloading Sentinel-1 data from GEE for {acquisition_time}...")
    download_url = image_export.getDownloadURL({
        'scale': 10,  # 10 meters per pixel
        'crs': 'EPSG:4326',
        'region': aoi,
        'format': 'GEO_TIFF'
    })

    # 4. Download and extract the ZIP file from Google
    response = requests.get(download_url)
    if response.status_code != 200:
        raise Exception("Failed to download image from Google Earth Engine.")

    with zipfile.ZipFile(io.BytesBytesIO(response.content)) as z:
        # GEE zips the bands. We extract the downloaded GeoTIFF
        tif_filename = z.namelist()[0]
        extracted_path = os.path.join(output_dir, "latest_scene.tif")
        
        # Save to disk
        with open(extracted_path, 'wb') as f:
            f.write(z.read(tif_filename))

    print(f"✅ Successfully downloaded SAR scene to {extracted_path}")

    return {
        "filepath": extracted_path,
        "acquisition_time": acquisition_time,
        "latitude": lat,
        "longitude": lon
    }