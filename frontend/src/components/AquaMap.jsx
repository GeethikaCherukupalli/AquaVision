import { Fragment } from 'react';
import { CircleMarker, MapContainer, Polygon, Polyline, Popup, TileLayer } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

function originPolygon(analysis) {
  const feature = analysis?.stage_2?.origin_region?.features?.find(
    (item) => item.geometry?.type === 'Polygon'
  );
  return feature?.geometry?.coordinates?.[0]?.map(([lon, lat]) => [lat, lon]) || [];
}

function vesselData(analysis) {
  return analysis?.stage_3?.vessel_candidates?.map((vessel) => ({
    ...vessel,
    positions: vessel.positions || [],
  })) || [];
}

function forecastPoints(analysis) {
  return analysis?.stage_2?.forecast?.points?.map((point) => [point.lat, point.lon]) || [];
}

function mapCenter(analysis) {
  const polygon = originPolygon(analysis);
  if (polygon.length) {
    const lat = polygon.reduce((sum, point) => sum + point[0], 0) / polygon.length;
    const lon = polygon.reduce((sum, point) => sum + point[1], 0) / polygon.length;
    return [lat, lon];
  }
  return [20, 0];
}

export default function AquaMap({ analysis, onSelectVessel }) {
  const polygon = originPolygon(analysis);
  const vessels = vesselData(analysis);
  const forecast = forecastPoints(analysis);

  return (
    <div className="map-container">
      <MapContainer center={mapCenter(analysis)} zoom={analysis ? 7 : 2} zoomControl style={{ width: '100%', height: '100%' }}>
        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {polygon.length > 2 && (
          <Polygon positions={polygon} pathOptions={{ color: '#7ef1c4', fillColor: '#7ef1c4', fillOpacity: 0.2, weight: 2 }}>
            <Popup>Probable origin region</Popup>
          </Polygon>
        )}
        {forecast.length > 1 && (
          <Polyline positions={forecast} pathOptions={{ color: '#f59e0b', weight: 3, dashArray: '8 8' }}>
            <Popup>Forecast trajectory</Popup>
          </Polyline>
        )}
        {vessels.map((vessel) => {
          const latest = vessel.positions[vessel.positions.length - 1];
          if (!latest) return null;
          const track = vessel.positions.map((point) => [point.lat, point.lon]);
          return (
            <Fragment key={vessel.vessel_id}>
              {track.length > 1 && <Polyline positions={track} pathOptions={{ color: '#7dd3fc', weight: 2 }} />}
              <CircleMarker
                center={[latest.lat, latest.lon]}
                radius={8}
                pathOptions={{ color: vessel.classification === 'high' ? '#f59e0b' : '#38bdf8', fillOpacity: 0.85 }}
                eventHandlers={{ click: () => onSelectVessel?.(vessel.vessel_id) }}
              >
                <Popup>
                  <strong>Vessel of interest</strong><br />
                  {vessel.vessel_id}<br />
                  Score: {vessel.score ?? 'Unavailable'}
                </Popup>
              </CircleMarker>
            </Fragment>
          );
        })}
      </MapContainer>
      <div className="map-label">2D OPERATIONAL MAP</div>
      {!analysis && <div className="map-empty">No geospatial analysis data</div>}
      {analysis?.demo_mode && <div className="map-demo-label">DEMO DATA</div>}
    </div>
  );
}
