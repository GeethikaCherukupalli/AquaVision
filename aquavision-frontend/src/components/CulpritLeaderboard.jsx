import React from 'react';
import { ShieldAlert, ShieldCheck, AlertTriangle, Navigation, ExternalLink } from 'lucide-react';

export default function CulpritLeaderboard({ suspects, onSelectVessel }) {
  if (!suspects || suspects.length === 0) {
    return (
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 text-center text-slate-500 text-xs font-mono">
        NO VESSEL TRAFFIC CORRELATED.
      </div>
    );
  }

  // Restrict strictly to the Top 3 suspects to avoid clutter
  const topSuspects = suspects.slice(0, 3);

  const getTierBadge = (tier) => {
    switch (tier) {
      case 'red':
        return <span className="flex items-center gap-1 text-[10px] font-bold text-rose-400 bg-rose-950/60 border border-rose-900/50 px-2 py-0.5 rounded"><ShieldAlert className="w-3 h-3" /> PROBABLE SUSPECT</span>;
      case 'yellow':
        return <span className="flex items-center gap-1 text-[10px] font-bold text-amber-400 bg-amber-950/60 border border-amber-900/50 px-2 py-0.5 rounded"><AlertTriangle className="w-3 h-3" /> ANOMALOUS</span>;
      default:
        return <span className="flex items-center gap-1 text-[10px] font-bold text-emerald-400 bg-emerald-950/60 border border-emerald-900/50 px-2 py-0.5 rounded"><ShieldCheck className="w-3 h-3" /> CLEAR</span>;
    }
  };

  return (
    <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 shadow-xl flex flex-col mt-3">
      <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-2">
        <h3 className="text-[10px] uppercase font-bold text-slate-400 tracking-widest flex items-center gap-1.5">
          <Navigation className="w-3.5 h-3.5 text-blue-400" /> Top Suspect Attribution (AI Ranked)
        </h3>
        <span className="text-[9px] font-mono text-slate-500">FILTERED TRAFFIC</span>
      </div>

      <div className="space-y-2.5">
        {topSuspects.map((vessel, index) => (
          <div 
            key={vessel.mmsi}
            onClick={() => onSelectVessel && onSelectVessel(vessel)}
            className="bg-slate-950/60 hover:bg-slate-800/50 border border-slate-800/80 rounded-lg p-3 transition cursor-pointer group relative"
          >
            <div className="flex items-center justify-between mb-1.5">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono font-bold text-slate-500">0{index + 1}</span>
                <span className="text-xs font-semibold text-slate-200 group-hover:text-cyan-400 transition">
                  {vessel.name}
                </span>
              </div>
              {getTierBadge(vessel.risk_tier)}
            </div>

            <div className="grid grid-cols-3 gap-2 text-[10px] font-mono text-slate-400 mb-2">
              <div>MMSI: <span className="text-slate-200">{vessel.mmsi}</span></div>
              <div>Dist: <span className="text-slate-200">{vessel.distance_km} km</span></div>
              <div>Score: <span className="text-cyan-400 font-bold">{vessel.composite_score}/100</span></div>
            </div>

            {/* Score Breakdown Bar Indicator */}
            <div className="w-full bg-slate-900 h-1.5 rounded-full overflow-hidden flex">
              <div style={{ width: `${vessel.composite_score}%` }} className={`h-full ${vessel.risk_tier === 'red' ? 'bg-rose-500' : vessel.risk_tier === 'yellow' ? 'bg-amber-500' : 'bg-emerald-500'}`}></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}