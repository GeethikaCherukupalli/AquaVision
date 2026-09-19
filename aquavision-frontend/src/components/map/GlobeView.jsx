import { useEffect, useRef } from 'react';
import Globe from 'globe.gl';

export default function GlobeView({ analysis, aoi }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return undefined;
    const globe = Globe()(ref.current).backgroundColor('#07121b').globeImageUrl('https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg').bumpImageUrl('https://unpkg.com/three-globe/example/img/earth-topology.png').showAtmosphere(true).atmosphereColor('#63c2d8').atmosphereAltitude(0.08);
    globe.pointsData([]).pathsData([]);
    const center = aoi ? { lat: (aoi[1] + aoi[3]) / 2, lng: (aoi[0] + aoi[2]) / 2, altitude: 1.8 } : { lat: 20, lng: 0, altitude: 2.4 };
    globe.pointOfView(center, 0); globe.controls().autoRotate = false;
    const resize = () => globe.width(ref.current.clientWidth).height(ref.current.clientHeight); resize(); window.addEventListener('resize', resize);
    return () => { window.removeEventListener('resize', resize); globe._destructor?.(); };
  }, [analysis, aoi]);
  return <div className="globe-frame"><div ref={ref} className="globe-canvas" /><div className="map-stamp"><strong>3D EARTH VIEW</strong><span>{aoi ? 'SELECTED AOI CONTEXT' : 'NO ORBITAL POSITION AVAILABLE'}</span></div></div>;
}