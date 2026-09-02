import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polygon, Polyline, ZoomControl, useMap, useMapEvents } from 'react-leaflet';
import { Search, MapPin, Hexagon, Ruler, Layers, CheckCircle, Trash2 } from 'lucide-react';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

const fetchLocation = async (query) => {
  try {
    const res = await fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}`);
    const data = await res.json();
    return data.length > 0 ? { lat: parseFloat(data[0].lat), lon: parseFloat(data[0].lon) } : null;
  } catch (err) {
    console.error("Geocoding failed", err);
    return null;
  }
};

const calculateDistance = (lat1, lon1, lat2, lon2) => {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) + Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) * Math.sin(dLon / 2) * Math.sin(dLon / 2);
  return R * (2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a)));
};

function ChangeView({ center }) {
  const map = useMap();
  useEffect(() => {
    if (center && center[0] !== undefined && center[1] !== undefined) {
      map.setView(center, map.getZoom(), { animate: true });
    }
  }, [center[0], center[1]]);
  return null;
}

const targetIcon = new L.DivIcon({
  className: 'custom-target-pin',
  html: `<div style="background-color: #06b6d4; width: 16px; height: 16px; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 12px #06b6d4; animation: pulse 2s infinite;"></div>`,
  iconSize: [16, 16],
  iconAnchor: [8, 8],
});

const vertexIcon = new L.DivIcon({
  className: 'custom-vertex-pin',
  html: `<div style="background-color: white; width: 12px; height: 12px; border-radius: 50%; border: 2px solid #eab308; box-shadow: 0 0 8px #eab308;"></div>`,
  iconSize: [12, 12],
  iconAnchor: [6, 6],
});

function ClickHandler({ onMapClick, activeTool }) {
  useMapEvents({
    click(e) {
      if (activeTool === 'point' || activeTool === 'polygon') {
        onMapClick(e.latlng.lat, e.latlng.lng);
      }
    },
  });
  return null;
}

export default function MapViewer({ targetLat, targetLon, detection, onLocationSelect }) {
  const [activeTool, setActiveTool] = useState('point'); 
  const [searchQuery, setSearchQuery] = useState('');
  const [draftPolygon, setDraftPolygon] = useState([]);
  const [perimeterKm, setPerimeterKm] = useState(0);

  // Safely handle empty strings on load by providing a default center
  const activeCenter = detection?.spatial_features?.centroid
    ? [detection.spatial_features.centroid.latitude, detection.spatial_features.centroid.longitude]
    : (targetLat !== '' && targetLon !== '' ? [targetLat, targetLon] : [25.0, -90.0]);

  const handleMapClick = (lat, lon) => {
    if (activeTool === 'point') {
      setDraftPolygon([]); 
      if (onLocationSelect) onLocationSelect(lat, lon);
    } else if (activeTool === 'polygon') {
      const newPoints = [...draftPolygon, [lat, lon]];
      setDraftPolygon(newPoints);
      if (newPoints.length > 1) {
        let dist = 0;
        for (let i = 0; i < newPoints.length - 1; i++) {
          dist += calculateDistance(newPoints[i][0], newPoints[i][1], newPoints[i+1][0], newPoints[i+1][1]);
        }
        setPerimeterKm(dist.toFixed(2));
      }
    }
  };

  const handleSearch = async (e) => {
    if (e.key === 'Enter') {
      const coords = await fetchLocation(searchQuery);
      if (coords && onLocationSelect) {
        setActiveTool('point');
        setDraftPolygon([]);
        onLocationSelect(coords.lat, coords.lon);
      }
    }
  };

  const handleFinishPolygon = () => {
    if (draftPolygon.length < 3) return;
    const avgLat = draftPolygon.reduce((sum, p) => sum + p[0], 0) / draftPolygon.length;
    const avgLon = draftPolygon.reduce((sum, p) => sum + p[1], 0) / draftPolygon.length;
    if (onLocationSelect) onLocationSelect(avgLat, avgLon);
    setActiveTool('point'); 
  };

  const handleClearPolygon = () => {
    setDraftPolygon([]);
    setPerimeterKm(0);
  };

  return (
    <div className="w-full h-full relative">
      <MapContainer 
        center={activeCenter} 
        zoom={5} 
        minZoom={3} 
        maxBounds={[[-90, -180], [90, 180]]}
        zoomControl={false} 
        className="h-full w-full bg-slate-950 z-0"
      >
        <ZoomControl position="bottomright" />
        <ChangeView center={activeCenter} />
        <ClickHandler onMapClick={handleMapClick} activeTool={activeTool} />

        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
          attribution='&copy; Esri &mdash; Sentinel-1 SAR'
          maxZoom={16}
          noWrap={true}
          bounds={[[-90, -180], [90, 180]]}
        />

        {/* Standard Point Selection - Only render if coordinates are valid numbers */}
        {activeTool === 'point' && draftPolygon.length === 0 && targetLat !== '' && targetLon !== '' && (
          <Marker position={[targetLat, targetLon]} icon={targetIcon} />
        )}

        {/* Interactive Polygon Drawing */}
        {draftPolygon.map((pos, idx) => <Marker key={idx} position={pos} icon={vertexIcon} />)}
        
        {draftPolygon.length === 2 && (
          <Polyline positions={draftPolygon} pathOptions={{ color: '#eab308', weight: 2, dashArray: '5, 5' }} />
        )}
        
        {draftPolygon.length > 2 && (
          <Polygon 
            positions={draftPolygon} 
            pathOptions={{ color: '#eab308', fillColor: '#eab308', fillOpacity: 0.2, weight: 2, dashArray: '5, 5' }} 
          />
        )}

        {/* AI Detection Polygon (From Backend) */}
        {detection?.spatial_features?.geometry_geojson && (
          <Polygon
            positions={detection.spatial_features.geometry_geojson.coordinates[0].map((c) => [c[1], c[0]])}
            pathOptions={{ color: '#f43f5e', fillColor: '#e11d48', fillOpacity: 0.65, weight: 2 }}
          />
        )}
      </MapContainer>

      {/* Instructional HUD */}
      <div className="absolute top-4 left-4 bg-slate-900/95 border border-slate-700 rounded p-3 z-[1000] text-xs shadow-xl text-slate-200 w-72">
        <div className="flex items-center gap-2 mb-1.5 border-b border-slate-700/50 pb-1.5">
          <div className="w-2 h-2 rounded-full bg-blue-500"></div>
          <span className="font-semibold text-slate-300 uppercase tracking-wide text-[10px]">Copernicus GEE Link</span>
        </div>
        <p className="text-slate-400 text-[11px] leading-relaxed mb-2">
          Use the right toolbar to search a location, drop a target point, or draw an Area of Interest (AOI).
        </p>
      </div>

      {/* Measurement Tooltip */}
      {activeTool === 'polygon' && draftPolygon.length > 1 && (
        <div className="absolute top-20 right-20 bg-slate-900 border border-slate-700 text-slate-200 font-bold px-3 py-1.5 rounded shadow-lg z-[1000] flex items-center gap-2 text-xs">
          <Ruler className="w-3.5 h-3.5 text-cyan-400" />
          <span>{perimeterKm} km</span>
        </div>
      )}

      {/* AOI Action Buttons */}
      {activeTool === 'polygon' && draftPolygon.length > 0 && (
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-[1000] flex gap-2">
          <button onClick={handleClearPolygon} className="bg-slate-900 hover:bg-slate-800 text-rose-400 px-4 py-2 rounded-full font-bold shadow-xl border border-rose-900/50 flex items-center gap-2 text-xs transition">
            <Trash2 className="w-4 h-4" /> Clear
          </button>
          {draftPolygon.length > 2 && (
            <button onClick={handleFinishPolygon} className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-full font-bold shadow-xl border border-blue-400 flex items-center gap-2 text-xs transition">
              <CheckCircle className="w-4 h-4" /> Finalize AOI
            </button>
          )}
        </div>
      )}

      {/* Top Right Toolbar (Enlarged) */}
      <div className="absolute top-4 right-4 z-[1000] flex flex-col items-end gap-3">
        {/* Search Bar */}
        <div className="bg-slate-900 border border-slate-700 rounded-lg shadow-xl flex items-center p-3">
          <Search className="w-5 h-5 text-slate-400 mr-2" />
          <input 
            type="text" 
            placeholder="Search Place (Press Enter)" 
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={handleSearch}
            className="bg-transparent border-none text-sm text-white focus:outline-none w-52 placeholder:text-slate-500" 
          />
        </div>

        {/* Tools */}
        <div className="bg-slate-900 border border-slate-700 rounded-lg shadow-xl flex flex-col w-12 overflow-hidden">
          <button onClick={() => { setActiveTool('point'); setDraftPolygon([]); }} title="Point Selection" className={`p-3 transition-colors border-b border-slate-800 ${activeTool === 'point' ? 'bg-blue-600/20 border-l-2 border-l-blue-500 text-blue-400' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'}`}>
            <MapPin className="w-5 h-5 mx-auto" />
          </button>
          <button onClick={() => { setActiveTool('polygon'); handleClearPolygon(); }} title="Draw AOI Polygon" className={`p-3 transition-colors border-b border-slate-800 ${activeTool === 'polygon' ? 'bg-blue-600/20 border-l-2 border-l-blue-500 text-blue-400' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'}`}>
            <Hexagon className="w-5 h-5 mx-auto" />
          </button>
          <button title="Toggle Overlays" className="p-3 hover:bg-slate-800 transition-colors border-b border-slate-800 text-slate-400 hover:text-slate-200">
            <Layers className="w-5 h-5 mx-auto" />
          </button>
          <button title="Measurement Mode" className="p-3 hover:bg-slate-800 transition-colors text-slate-400 hover:text-slate-200">
            <Ruler className="w-5 h-5 mx-auto" />
          </button>
        </div>
      </div>
    </div>
  );
}