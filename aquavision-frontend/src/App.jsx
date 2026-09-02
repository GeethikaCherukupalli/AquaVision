import React, { useState } from 'react';
import { Play, RotateCw, Calendar, Crosshair, Anchor, XCircle } from 'lucide-react';
import { runFullInvestigation } from './api/client';
import MapViewer from './components/MapViewer';
import SpillSummary from './components/SpillSummary';
import CulpritLeaderboard from './components/CulpritLeaderboard';
import MetoceanSimulator from './components/MetoceanSimulator';
import AnalyticsModal from './components/AnalyticsModal';

export default function App() {
  const [data, setData] = useState(null);
  const [selectedVessel, setSelectedVessel] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');

  // Initialized to empty strings so the map is blank on load
  const [lat, setLat] = useState('');
  const [lon, setLon] = useState('');
  const [targetDate, setTargetDate] = useState('');

  const executePipeline = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await runFullInvestigation(lat, lon, targetDate || null);
      setData(result);
    } catch (err) {
      console.error(err);
      setError('Connection refused: Backend server is offline or unreachable.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setLat('');
    setLon('');
    setTargetDate('');
    setData(null);
    setError(null);
  };

  const handleMapClick = (clickedLat, clickedLon) => {
    setLat(parseFloat(clickedLat.toFixed(4)));
    setLon(parseFloat(clickedLon.toFixed(4)));
  };

  return (
    <div className="flex h-screen w-screen bg-slate-950 text-slate-100 font-sans overflow-hidden">
      <div className="w-[440px] bg-slate-950 border-r border-slate-800/80 flex flex-col p-4 z-20 shadow-2xl overflow-y-auto">
        
        <div className="flex items-center justify-between mb-5 border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <Anchor className="w-5 h-5 text-slate-400" />
            <h1 className="text-lg font-semibold tracking-widest text-slate-200">AQUAVISION</h1>
          </div>
          <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-slate-900 text-slate-400 border border-slate-700">
            MARITIME SURVEILLANCE
          </span>
        </div>

        <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800 mb-4 space-y-4 text-sm">
          <div className="flex items-center justify-between text-slate-400">
            <span className="flex items-center gap-1.5"><Crosshair className="w-4 h-4 text-cyan-500" /> Lat / Lon:</span>
            <div className="flex gap-2">
              <input 
                type="number" 
                value={lat} 
                onChange={(e) => setLat(e.target.value === '' ? '' : parseFloat(e.target.value))}
                className="w-24 bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-cyan-400 font-mono text-center focus:outline-none focus:border-cyan-500 text-sm shadow-inner"
                step="0.0001"
              />
              <input 
                type="number" 
                value={lon} 
                onChange={(e) => setLon(e.target.value === '' ? '' : parseFloat(e.target.value))}
                className="w-24 bg-slate-950 border border-slate-700 rounded px-2 py-1.5 text-cyan-400 font-mono text-center focus:outline-none focus:border-cyan-500 text-sm shadow-inner"
                step="0.0001"
              />
            </div>
          </div>
          <div className="flex items-center justify-between text-slate-400">
            <span className="flex items-center gap-1.5"><Calendar className="w-4 h-4 text-slate-500" /> Target Date:</span>
            <input 
              type="date" 
              value={targetDate} 
              onChange={(e) => setTargetDate(e.target.value)}
              className="bg-slate-950 border border-slate-700 rounded px-3 py-1.5 text-slate-200 font-mono text-xs focus:outline-none focus:border-cyan-500 shadow-inner w-[192px]"
            />
          </div>
        </div>

        {/* Action Buttons: Query & Reset */}
        <div className="flex gap-2 mb-4">
          <button
            onClick={executePipeline}
            disabled={loading || lat === '' || lon === ''}
            className="flex-1 py-2.5 px-4 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 disabled:text-slate-500 text-white font-semibold text-xs rounded-lg shadow-lg flex items-center justify-center gap-2 transition-colors border border-blue-500/50"
          >
            {loading ? <RotateCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-current" />}
            <span>{loading ? 'Executing...' : 'Query Location'}</span>
          </button>
          <button
            onClick={handleReset}
            disabled={loading}
            className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold text-xs rounded-lg border border-slate-700 transition-colors flex items-center gap-1.5"
          >
            <XCircle className="w-4 h-4" /> Reset
          </button>
        </div>

        {error && <div className="p-3 mb-3 bg-rose-950/30 border border-rose-900/50 text-rose-400 text-xs rounded-lg font-mono">{error}</div>}

        <div className="flex border-b border-slate-800 mb-3 text-xs">
          <button onClick={() => setActiveTab('overview')} className={`flex-1 py-2 font-semibold border-b-2 transition ${activeTab === 'overview' ? 'border-blue-500 text-slate-200' : 'border-transparent text-slate-500 hover:text-slate-400'}`}>Spill Intelligence</button>
          <button onClick={() => setActiveTab('simulation')} className={`flex-1 py-2 font-semibold border-b-2 transition ${activeTab === 'simulation' ? 'border-blue-500 text-slate-200' : 'border-transparent text-slate-500 hover:text-slate-400'}`}>Metocean Simulator</button>
        </div>

        {data && (
          <div className="flex-1 flex flex-col min-h-0">
            {activeTab === 'overview' ? (
              <><SpillSummary detection={data.detection} onOpenDiagnostics={() => setIsModalOpen(true)} /><CulpritLeaderboard suspects={data.suspect_ranking} /></>
            ) : (<MetoceanSimulator hindcastData={data.metocean_hindcast} />)}
          </div>
        )}
      </div>

      <div className="flex-1 h-full relative">
        <MapViewer
          targetLat={lat}
          targetLon={lon}
          detection={data?.detection}
          onLocationSelect={handleMapClick}
        />
      </div>
      <AnalyticsModal sarData={data?.detection} isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} />
    </div>
  );
}