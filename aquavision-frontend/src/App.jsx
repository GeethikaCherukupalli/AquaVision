import React, { useState, useEffect } from 'react';
import { Play, RotateCw, Radio, Calendar, MapPin } from 'lucide-react';
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

  // Interactive coordinate & date states
  const [lat, setLat] = useState(28.932);
  const [lon, setLon] = useState(-88.974);
  const [targetDate, setTargetDate] = useState('');

  const executePipeline = async (customLat = lat, customLon = lon, customDate = targetDate) => {
    setLoading(true);
    setError(null);
    try {
      const result = await runFullInvestigation(customLat, customLon, customDate || null);
      setData(result);
    } catch (err) {
      console.error(err);
      setError('Could not connect to FastAPI server or fetch GEE data.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    executePipeline(28.932, -88.974, '');
  }, []);

  const handleMapClick = (clickedLat, clickedLon) => {
    setLat(clickedLat);
    setLon(clickedLon);
    executePipeline(clickedLat, clickedLon, targetDate);
  };

  return (
    <div className="flex h-screen w-screen bg-slate-950 text-slate-100 font-sans overflow-hidden">
      {/* Sidebar Controls */}
      <div className="w-[440px] bg-slate-950 border-r border-slate-800/80 flex flex-col p-4 z-20 shadow-2xl overflow-y-auto">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Radio className="w-5 h-5 text-cyan-400 animate-pulse" />
            <h1 className="text-lg font-black tracking-widest text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">
              AQUAVISION AI
            </h1>
          </div>
          <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
            SIH LIVE ENGINE
          </span>
        </div>

        {/* Coordinate & Date Controls */}
        <div className="bg-slate-900/60 p-3 rounded-xl border border-slate-800 mb-3 space-y-2 text-xs">
          <div className="flex items-center justify-between text-slate-400">
            <span className="flex items-center gap-1"><MapPin className="w-3.5 h-3.5 text-cyan-400" /> Target Coordinates:</span>
            <span className="font-mono text-cyan-300">{lat.toFixed(4)}°, {lon.toFixed(4)}°</span>
          </div>
          <div className="flex items-center justify-between text-slate-400">
            <span className="flex items-center gap-1"><Calendar className="w-3.5 h-3.5 text-amber-400" /> Incident Date:</span>
            <input 
              type="date" 
              value={targetDate} 
              onChange={(e) => setTargetDate(e.target.value)}
              className="bg-slate-950 border border-slate-700 rounded px-2 py-1 text-slate-200 font-mono text-[11px]"
            />
          </div>
        </div>

        <button
          onClick={() => executePipeline(lat, lon, targetDate)}
          disabled={loading}
          className="w-full py-2.5 px-4 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 disabled:opacity-50 text-white font-bold text-xs rounded-xl shadow-lg shadow-cyan-950 flex items-center justify-center gap-2 mb-3 transition"
        >
          {loading ? <RotateCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-current" />}
          <span>{loading ? 'Fetching GEE Satellite Pass & Running AI...' : 'Query Location & Run Pipeline'}</span>
        </button>

        {error && (
          <div className="p-3 mb-3 bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs rounded-xl">
            {error}
          </div>
        )}

        {/* Tab Switcher */}
        <div className="flex border-b border-slate-800 mb-3 text-xs">
          <button
            onClick={() => setActiveTab('overview')}
            className={`flex-1 py-2 font-bold border-b-2 transition ${activeTab === 'overview' ? 'border-cyan-400 text-cyan-300' : 'border-transparent text-slate-500 hover:text-slate-300'}`}
          >
            Spill Intelligence
          </button>
          <button
            onClick={() => setActiveTab('simulation')}
            className={`flex-1 py-2 font-bold border-b-2 transition ${activeTab === 'simulation' ? 'border-cyan-400 text-cyan-300' : 'border-transparent text-slate-500 hover:text-slate-300'}`}
          >
            Metocean Simulator
          </button>
        </div>

        {data && (
          <div className="flex-1 flex flex-col min-h-0">
            {activeTab === 'overview' ? (
              <>
                <SpillSummary detection={data.detection} onOpenDiagnostics={() => setIsModalOpen(true)} />
                <CulpritLeaderboard suspects={data.suspect_ranking} />
              </>
            ) : (
              <MetoceanSimulator hindcastData={data.metocean_hindcast} />
            )}
          </div>
        )}
      </div>

      {/* Map View */}
      <div className="flex-1 h-full relative">
        <MapViewer
          detection={data?.detection}
          hindcast={data?.metocean_hindcast}
          suspects={data?.suspect_ranking}
          selectedVessel={selectedVessel}
          onSelectVessel={setSelectedVessel}
          onLocationSelect={handleMapClick}
        />
      </div>

      <AnalyticsModal
        sarData={data?.detection}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
      />
    </div>
  );
}