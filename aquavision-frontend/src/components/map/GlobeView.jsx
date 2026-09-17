import { useEffect, useRef } from 'react';
import Globe from 'globe.gl';

export default function GlobeView({ analysis }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return undefined;
    const globe = Globe()(ref.current).backgroundColor('#07121b').globeImageUrl('https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg').bumpImageUrl('https://unpkg.com/three-globe/example/img/earth-topology.png').showAtmosphere(true).atmosphereColor('#63c2d8').atmosphereAltitude(0.08);
    const points = analysis?.stage_3?.vessel_candidates?.flatMap((vessel) => { const point = vessel.positions?.at(-1); return point ? [{ lat: point.lat, lng: point.lon, label: vessel.vessel_id, color: '#63c2d8' }] : []; }) || [];
    const forecast = analysis?.stage_2?.forecast?.points || [];
    globe.pointsData(points).pointColor((point) => point.color).pointLabel((point) => point.label).pointRadius(0.035).pathsData(forecast.length > 1 ? [{ label: 'Forecast', points: forecast.map((point) => [point.lat, point.lon]) }] : []).pathPointLat((point) => point[0]).pathPointLng((point) => point[1]).pathColor(() => '#f2b84b').pathDashLength(0.35).pathDashGap(0.12);
    globe.pointOfView({ lat: 16, lng: 72, altitude: 2.1 }, 0); globe.controls().autoRotate = false;
    const resize = () => globe.width(ref.current.clientWidth).height(ref.current.clientHeight); resize(); window.addEventListener('resize', resize);
    return () => { window.removeEventListener('resize', resize); globe._destructor?.(); };
  }, [analysis]);
  return <div className="globe-frame"><div ref={ref} className="globe-canvas" /><div className="map-stamp"><strong>3D SITUATIONAL GLOBE</strong><span>SENTINEL ORBIT / VESSEL CONTEXT</span></div></div>;
}