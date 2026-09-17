const items = [['Overview', '01'], ['Map', '02'], ['Analysis', '03'], ['Forecast', '04'], ['AIS', '05'], ['Scenes', '06'], ['Reports', '07']];

export default function Sidebar({ active, setActive }) {
  return <aside className="sidebar"><div className="rail-title">WORKSPACE</div><nav>{items.map(([label, number]) => <button key={label} className={active === label ? 'selected' : ''} onClick={() => setActive(label)}><span>{number}</span>{label}</button>)}</nav><div className="rail-footer"><span className="status-dot green" /> SERVICES<br /><small>API v0.1.0</small></div></aside>;
}