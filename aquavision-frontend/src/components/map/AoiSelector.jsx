import { useRef, useState } from 'react';
import { Rectangle, useMapEvents } from 'react-leaflet';

export default function AoiSelector({ aoi, onChange }) {
  const startRef = useRef(null);
  const [draft, setDraft] = useState(null);

  useMapEvents({
    mousedown(event) {
      if (event.originalEvent.button !== 0) return;
      startRef.current = event.latlng;
      setDraft([event.latlng, event.latlng]);
      event.target.dragging.disable();
    },
    mousemove(event) {
      if (!startRef.current) return;
      setDraft([startRef.current, event.latlng]);
    },
    mouseup(event) {
      if (!startRef.current) return;
      const start = startRef.current;
      startRef.current = null;
      event.target.dragging.enable();
      const south = Math.min(start.lat, event.latlng.lat);
      const north = Math.max(start.lat, event.latlng.lat);
      const west = Math.min(start.lng, event.latlng.lng);
      const east = Math.max(start.lng, event.latlng.lng);
      if (east - west < 0.05 || north - south < 0.05) {
        setDraft(null);
        return;
      }
      setDraft(null);
      onChange([west, south, east, north]);
    },
  });

  const positions = draft || (aoi ? [[aoi[1], aoi[0]], [aoi[3], aoi[2]]] : null);
  return positions ? <Rectangle bounds={positions} pathOptions={{ color: '#f2b84b', weight: 2, fillColor: '#f2b84b', fillOpacity: 0.14 }} /> : null;
}
