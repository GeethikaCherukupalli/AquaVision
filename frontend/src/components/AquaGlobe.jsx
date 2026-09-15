import { useEffect, useRef, useState } from 'react';
import Globe from 'globe.gl';

function polygonFeatures(analysis) {
  const featureCollection = analysis?.stage_2?.origin_region;
  return featureCollection?.features?.flatMap((feature) => {
    if (feature.geometry?.type !== 'Polygon') return [];
    return [{ points: feature.geometry.coordinates[0].map(([lng, lat]) => ({ lat, lng })) }];
  }) || [];
}

function vesselPoints(analysis) {
  return analysis?.stage_3?.vessel_candidates?.flatMap((vessel) => {
    const positions = vessel.positions || [];
    return positions.length ? [{
      lat: positions[positions.length - 1].lat,
      lng: positions[positions.length - 1].lon,
      label: vessel.vessel_id,
      color: vessel.classification === 'high' ? '#f59e0b' : '#7dd3fc',
    }] : [];
  }) || [];
}

function vesselPaths(analysis) {
  return analysis?.stage_3?.vessel_candidates?.flatMap((vessel) => {
    const points = vessel.positions || [];
    return points.length > 1 ? [{ label: vessel.vessel_id, points: points.map((point) => [point.lat, point.lon]) }] : [];
  }) || [];
}

function forecastPaths(analysis) {
  const points = analysis?.stage_2?.forecast?.points || [];
  return points.length > 1 ? [{ label: 'Forecast trajectory', points: points.map((point) => [point.lat, point.lon]) }] : [];
}

export default function AquaGlobe({ analysis, onSelectVessel }) {
  const containerRef = useRef(null);
  const globeRef = useRef(null);
  const [status, setStatus] = useState('INITIALIZING');

  useEffect(() => {
    if (!containerRef.current) return undefined;
    const globe = Globe()(containerRef.current)
      .backgroundColor('#06131d')
      .globeImageUrl('https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
      .bumpImageUrl('https://unpkg.com/three-globe/example/img/earth-topology.png')
      .showAtmosphere(true)
      .atmosphereColor('#57d7ff')
      .atmosphereAltitude(0.12)
      .pointLat((point) => point.lat)
      .pointLng((point) => point.lng)
      .pointAltitude(0.02)
      .pointRadius(0.035)
      .pointColor((point) => point.color)
      .pointLabel((point) => point.label)
      .polygonCapColor(() => 'rgba(126, 241, 196, 0.16)')
      .polygonSideColor(() => 'rgba(126, 241, 196, 0.12)')
      .polygonStrokeColor(() => 'rgba(126, 241, 196, 0.7)')
      .polygonAltitude(0.01)
      .pathPointLat((point) => point[0])
      .pathPointLng((point) => point[1])
      .pathPointAlt(() => 0.02)
      .pathColor((path) => path.label === 'Forecast trajectory' ? '#f59e0b' : '#7dd3fc')
      .pathStroke(0.8)
      .pathDashLength(0.35)
      .pathDashGap(0.12)
      .pathDashAnimateTime(8000);

    globe.pointOfView({ lat: 20, lng: 70, altitude: 2.2 }, 0);
    globe.controls().autoRotate = false;
    globe.onPointClick((point) => {
      if (point.label) onSelectVessel?.(point.label);
    });
    globeRef.current = globe;
    setStatus('GLOBE ONLINE');

    const resize = () => globe.width(containerRef.current.clientWidth).height(containerRef.current.clientHeight);
    resize();
    window.addEventListener('resize', resize);
    return () => {
      window.removeEventListener('resize', resize);
      globe._destructor?.();
      globeRef.current = null;
    };
  }, []);

  useEffect(() => {
    const globe = globeRef.current;
    if (!globe) return;
    globe.pointsData(vesselPoints(analysis));
    globe.polygonsData(polygonFeatures(analysis));
    globe.pathsData([...vesselPaths(analysis), ...forecastPaths(analysis)]);
  }, [analysis]);

  return (
    <div className="globe-container">
      <div ref={containerRef} className="globe-canvas" />
      <div className="globe-status"><span className="status-dot online" />{status}</div>
      {!analysis && <div className="globe-empty">No geospatial analysis data</div>}
      {analysis?.demo_mode && <div className="globe-demo-label">DEMO DATA</div>}
      <div className="globe-legend">
        <span><i className="legend-vessel" />AIS vessel position</span>
        <span><i className="legend-origin" />Probable origin region</span>
        <span><i className="legend-forecast" />Forecast trajectory</span>
      </div>
    </div>
  );
}
