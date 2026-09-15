import { useEffect, useState } from 'react';
import AquaGlobe from './components/AquaGlobe';
import AquaMap from './components/AquaMap';

function App() {
  const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
  const [readiness, setReadiness] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState('historical');
  const [view, setView] = useState('globe');
  const [selectedVessel, setSelectedVessel] = useState(null);

  useEffect(() => {
    fetch(`${apiBase}/api/v1/readiness`)
      .then((response) => response.json())
      .then(setReadiness)
      .catch(() => setReadiness({ status: 'offline', missing_configuration: ['API unavailable'] }));
  }, [apiBase]);

  async function runAnalysis() {
    setLoading(true);
    try {
      const response = await fetch(`${apiBase}/api/v1/analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode, sensor: 'sentinel-1' }),
      });
      setAnalysis(await response.json());
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">AquaVision</div>
        <nav>
          <button className="nav active" type="button">Overview</button>
          <button className="nav" type="button">Scenes</button>
          <button className="nav" type="button">Forecast</button>
          <button className="nav" type="button">AIS</button>
        </nav>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div>
            <p className="eyebrow">Operations</p>
            <h1>Oil spill response dashboard</h1>
          </div>
          <div className="topbar-actions">
            <div className="mode-switch" role="group" aria-label="Analysis mode">
              <button type="button" className={mode === 'historical' ? 'mode active' : 'mode'} onClick={() => setMode('historical')}>Historical</button>
              <button type="button" className={mode === 'monitoring' ? 'mode active' : 'mode'} onClick={() => setMode('monitoring')}>Monitoring</button>
            </div>
            <div className="view-switch" role="group" aria-label="Map view">
              <button type="button" className={view === 'globe' ? 'mode active' : 'mode'} onClick={() => setView('globe')}>3D Globe</button>
              <button type="button" className={view === 'map' ? 'mode active' : 'mode'} onClick={() => setView('map')}>2D Map</button>
            </div>
            <button className="primary" onClick={runAnalysis} disabled={loading}>
            {loading ? 'Running...' : 'Run analysis'}
            </button>
          </div>
        </header>

        <div className={`status-banner ${readiness?.status === 'ready' ? 'ready' : ''}`}>
          <strong>{readiness?.status === 'ready' ? 'LIVE CONFIGURED' : 'DEMO MODE'}</strong>
          <span>
            {readiness?.status === 'ready'
              ? 'Production data and trained Stage 1 artifacts are available.'
              : 'Live providers and trained model artifacts are not configured.'}
          </span>
        </div>

        <section className="globe-card">
          {view === 'globe' ? (
            <AquaGlobe analysis={analysis?.result} onSelectVessel={setSelectedVessel} />
          ) : (
            <AquaMap analysis={analysis?.result} onSelectVessel={setSelectedVessel} />
          )}
        </section>

        {analysis?.result ? (
          <section className="metrics-grid">
            <Metric label="Spill area" value={`${analysis.result.stage_1?.geometry?.area_km2 ?? 'Unavailable'} km²`} />
            <Metric label="Confidence" value={formatPercent(analysis.result.stage_1?.quality?.confidence)} />
            <Metric label="Drift" value={analysis.result.stage_2 ? 'Available' : 'Unavailable'} />
            <Metric label="AIS matches" value={analysis.result.stage_3?.vessel_candidates?.length ?? 0} />
          </section>
        ) : (
          <div className="empty-state">No active analysis</div>
        )}

        <section className="content-grid">
          <div className="pane">
            <h2>Candidate vessels</h2>
            {analysis?.result?.stage_3?.vessel_candidates?.length ? <table>
              <thead>
                <tr>
                  <th>Vessel</th>
                  <th>Distance</th>
                  <th>Score</th>
                </tr>
              </thead>
              <tbody>
                {analysis.result.stage_3.vessel_candidates.map((row) => (
                  <tr key={row.vessel_id}>
                    <td>{row.vessel_id}</td>
                    <td>{row.minimum_distance_km ?? 'Unavailable'}</td>
                    <td>{row.score ?? 'Unavailable'}</td>
                  </tr>
                ))}
              </tbody>
            </table> : <p className="empty-copy">No AIS correlation available.</p>}
          </div>

          <div className="pane">
            <h2>Event summary</h2>
            {analysis?.result ? <ul className="summary-list">
              <li>{analysis.result.stage_1?.quality?.warnings?.join(' ') || 'Stage 1 result available.'}</li>
              <li>{analysis.result.stage_2 ? 'Drift or origin result available.' : 'Ocean forcing unavailable.'}</li>
              <li>{analysis.result.stage_3 ? analysis.result.stage_3.correlation_summary : 'AIS correlation unavailable.'}</li>
            </ul> : <p className="empty-copy">Run an analysis to populate incident evidence.</p>}
            {selectedVessel && <p className="selection-detail">Vessel of interest: {selectedVessel}</p>}
            {analysis?.job_id && <p className="run-id">Analysis {analysis.job_id}</p>}
            {readiness?.missing_configuration?.length > 0 && (
              <p className="missing">Missing: {readiness.missing_configuration.join(', ')}</p>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

function Metric({ label, value }) {
  return <div className="metric-card"><span>{label}</span><strong>{value}</strong></div>;
}

function formatPercent(value) {
  return typeof value === 'number' ? `${Math.round(value * 100)}%` : 'Unavailable';
}

export default App;
