import { MapContainer, TileLayer, useMap } from 'react-leaflet';
import { useEffect } from 'react';
import AoiSelector from './AoiSelector';
import 'leaflet/dist/leaflet.css';

function MapViewport({ aoi }) {
  const map = useMap();
  useEffect(() => {
    if (aoi) map.fitBounds([[aoi[1], aoi[0]], [aoi[3], aoi[2]]], { padding: [28, 28] });
  }, [aoi, map]);
  return null;
}

export default function MapView({ aoi, onAoiChange }) {
  const center = aoi ? [(aoi[1] + aoi[3]) / 2, (aoi[0] + aoi[2]) / 2] : [20, 0];
  return <div className="map-frame"><MapContainer center={center} zoom={aoi ? 7 : 2} worldCopyJump zoomControl>
    <TileLayer attribution="&copy; OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
    <MapViewport aoi={aoi} />
    <AoiSelector aoi={aoi} onChange={onAoiChange} />
  </MapContainer><div className="map-stamp"><strong>AOI SELECTION MAP</strong><span>DRAG TO DRAW A RECTANGLE</span></div><div className="map-readout">{aoi ? `BBOX ${aoi.map((value) => value.toFixed(3)).join(' / ')}` : 'NO AOI SELECTED'}</div><div className="map-scale">NAVIGATE, THEN DRAG TO SELECT</div></div>;
}