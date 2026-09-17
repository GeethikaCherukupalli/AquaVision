export default function Header({ mode, setMode, readiness }) {
  const isReady = readiness?.status === 'ready';
  return <header className="top-header">
    <div className="wordmark"><strong>AQUAVISION</strong><span>MARITIME INTELLIGENCE SYSTEM</span></div>
    <div className="mode-tabs" role="tablist" aria-label="Operational mode">
      {['historical', 'monitoring'].map((item) => <button key={item} className={mode === item ? 'active' : ''} onClick={() => setMode(item)}>{item}</button>)}
    </div>
    <div className="system-meta"><span className={`status-dot ${isReady ? 'green' : 'amber'}`} />{isReady ? 'SYSTEM ONLINE' : 'LIMITED AVAILABILITY'}<span className="meta-divider" />16 SEP 2026&nbsp; | &nbsp;22:42 IST</div>
  </header>;
}