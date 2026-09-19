import { useEffect, useState } from 'react';
import Header from './components/layout/Header';
import Sidebar from './components/layout/Sidebar';
import MapView from './components/map/MapView';
import GlobeView from './components/map/GlobeView';
import DetailsPanel from './components/analysis/DetailsPanel';
import { getReadiness, runAnalysis, searchAcquisitions } from './services/api';

function dateValue(date) {
  return date.toISOString().slice(0, 10);
}

const today = new Date();
const initialEndDate = dateValue(today);
const initialStartDate = dateValue(new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000));

export default function App() {
  const [mode, setMode] = useState('historical');
  const [active, setActive] = useState('Overview');
  const [view, setView] = useState('map');
  const [showAoi, setShowAoi] = useState(true);
  const [aoi, setAoi] = useState(null);
  const [startDate, setStartDate] = useState(initialStartDate);
  const [endDate, setEndDate] = useState(initialEndDate);
  const [readiness, setReadiness] = useState(null);
  const [products, setProducts] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    getReadiness().then(setReadiness).catch(() => setReadiness({ status: 'offline', providers: {} }));
  }, []);

  async function handleSearch() {
    if (!aoi) {
      setError('Select an AOI by dragging a rectangle on the map.');
      return;
    }
    setSearching(true);
    setError('');
    setProducts([]);
    setSelectedProduct(null);
    try {
      const response = await searchAcquisitions({
        region: { west: aoi[0], south: aoi[1], east: aoi[2], north: aoi[3] },
        start_date: startDate,
        end_date: endDate,
      });
      setProducts(response.results || []);
      if (!response.results?.length) setError('NO SENTINEL-1 ACQUISITIONS FOUND');
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setSearching(false);
    }
  }

  async function handleRun() {
    if (!aoi || !selectedProduct) {
      setError(!aoi ? 'Select an AOI on the map.' : 'Search CDSE and select a Sentinel-1 acquisition first.');
      return;
    }
    setLoading(true);
    setError('');
    setAnalysis(null);
    try {
      setAnalysis(await runAnalysis({
        mode,
        execution_mode: 'production',
        sensor: 'sentinel-1',
        region: { west: aoi[0], south: aoi[1], east: aoi[2], north: aoi[3] },
        start_date: startDate,
        end_date: endDate,
        acquisition_id: selectedProduct.id,
        acquisition_time: selectedProduct.acquisition_time,
      }));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  const data = analysis?.result;
  const detection = data?.candidate;
  const status = loading ? 'ANALYZING SENTINEL-1 IMAGERY' : error ? 'ANALYSIS FAILED / DATA UNAVAILABLE' : detection === true ? 'POTENTIAL OIL SPILL DETECTED' : detection === false ? 'NO OIL SPILL CANDIDATE DETECTED' : 'READY FOR ANALYSIS';

  function clearAoi() {
    setAoi(null);
    setProducts([]);
    setSelectedProduct(null);
    setAnalysis(null);
    setError('');
  }

  return <div className="app-shell">
    <Header mode={mode} setMode={setMode} readiness={readiness} />
    <Sidebar active={active} setActive={setActive} />
    <main className="workspace">
      <div className="workspace-toolbar"><div><span className="eyebrow">{active.toUpperCase()} / {mode.toUpperCase()}</span><h1>Sentinel-1 candidate detection</h1></div><div className="view-switch"><button className={view === 'map' ? 'active' : ''} onClick={() => setView('map')}>2D MAP</button><button className={view === 'globe' ? 'active' : ''} onClick={() => setView('globe')}>3D GLOBE</button></div></div>
      <section className={`detection-status ${loading ? 'loading' : error ? 'failed' : detection === true ? 'candidate' : detection === false ? 'clear' : 'ready'}`}><strong>ANALYSIS RESULT · {status}</strong><span>{data ? `Probability ${Math.round(data.classification_probability * 100)}% · Threshold ${Math.round(data.classification_threshold * 100)}% · ${data.model} ${data.model_version}` : 'Select an AOI, search the Copernicus catalogue, select an acquisition, then run Stage 1.'}</span></section>
      {error && <div className="error-banner">{error}</div>}
      <div className="workspace-grid">
        <div className="map-column"><div className="map-panel">{view === 'map' ? <MapView aoi={showAoi ? aoi : null} onAoiChange={setAoi} /> : <GlobeView analysis={data} aoi={aoi} />}{aoi && <button className="clear-aoi" onClick={clearAoi}>CLEAR AOI</button>}<div className="layer-control"><strong>MAP</strong><label><input type="checkbox" checked={showAoi} onChange={() => setShowAoi(!showAoi)} />Selected AOI</label></div></div><section className="panel acquisition-panel"><div className="panel-heading"><span>SENTINEL-1 ACQUISITIONS</span><small>{products.length ? `${products.length} FOUND` : 'SEARCH REQUIRED'}</small></div>{products.length ? products.map((product) => <button className={`acquisition-row ${selectedProduct?.id === product.id ? 'selected' : ''}`} key={product.id} onClick={() => setSelectedProduct(product)}><strong>{product.title || product.id}</strong><span>{product.acquisition_time || 'Acquisition time unavailable'} · {product.platform} · {product.orbit || 'Orbit unavailable'}</span></button>) : <p className="disclaimer">Search the CDSE catalogue for the selected AOI and dates.</p>}</section></div>
        <DetailsPanel mode={mode} aoi={aoi} startDate={startDate} endDate={endDate} setStartDate={setStartDate} setEndDate={setEndDate} selectedProduct={selectedProduct} onSearch={handleSearch} onRun={handleRun} searching={searching} loading={loading} readiness={readiness} analysis={analysis} />
      </div>
    </main>
  </div>;
}
