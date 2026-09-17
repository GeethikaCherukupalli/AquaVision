import { CircleMarker, MapContainer, Polygon, Polyline, TileLayer, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

export default function MapView({ analysis, layers }) {
  const geometry = analysis?.stage_1?.geometry;
  const center = geometry?.centroid ? [geometry.centroid.lat, geometry.centroid.lon] : [16.85, 72.25];
  const spill = geometry?.bbox ? [[geometry.bbox.min_lat, geometry.bbox.min_lon], [geometry.bbox.min_lat, geometry.bbox.max_lon], [geometry.bbox.max_lat, geometry.bbox.max_lon], [geometry.bbox.max_lat, geometry.bbox.min_lon]] : [];
  const origin = analysis?.stage_2?.origin_region?.features?.[0]?.geometry?.coordinates?.[0]?.map(([lon, lat]) => [lat, lon]) || [];
  const forecast = analysis?.stage_2?.forecast?.points?.map((point) => [point.lat, point.lon]) || [];
  const vessels = analysis?.stage_3?.vessel_candidates || [];
  return <div className="map-frame"><MapContainer center={center} zoom={analysis ? 7 : 5} zoomControl={false}>
    <TileLayer attribution="&copy; OpenStreetMap contributors" url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
    {layers.spill && spill.length > 2 && <Polygon positions={spill} pathOptions={{ color: '#e86b4a', fillColor: '#e86b4a', fillOpacity: 0.28, weight: 2 }}><Tooltip permanent direction="center">OIL SPILL · {Math.round((analysis.stage_1.quality.confidence || 0) * 100)}%</Tooltip></Polygon>}
    {layers.origin && origin.length > 2 && <Polygon positions={origin} pathOptions={{ color: '#f2b84b', fillColor: '#f2b84b', fillOpacity: 0.12, dashArray: '4 5', weight: 2 }} />}
    {layers.forecast && forecast.length > 1 && <Polyline positions={forecast} pathOptions={{ color: '#f2b84b', weight: 3, dashArray: '7 8' }} />}
    {layers.vessels && vessels.map((vessel) => { const latest = vessel.positions?.at(-1); if (!latest) return null; return <CircleMarker key={vessel.vessel_id} center={[latest.lat, latest.lon]} radius={5} pathOptions={{ color: '#63c2d8', fillColor: '#63c2d8', fillOpacity: 0.9 }}><Tooltip>{vessel.vessel_id} · association {vessel.classification || 'unknown'}</Tooltip></CircleMarker>; })}
  </MapContainer><div className="map-stamp"><strong>2D ANALYTICAL MAP</strong><span>SENTINEL-1 / ARABIAN SEA</span></div><div className="map-readout">LAT {center[0].toFixed(2)}° N<br />LON {center[1].toFixed(2)}° E</div><div className="map-scale">0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;50&nbsp;&nbsp;&nbsp;&nbsp;100 km</div>{!analysis && <div className="map-empty">Run an analysis to populate the geospatial workspace</div>}</div>;
}