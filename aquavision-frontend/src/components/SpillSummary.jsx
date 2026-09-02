import React from 'react';
import { AlertTriangle, Clock, Layers, CheckCircle, Activity, Ruler, Maximize } from 'lucide-react';

export default function SpillSummary({ detection }) {
  if (!detection) return null;

  const { geometric_properties, classification, artifacts } = detection;
  const isSpill = geometric_properties.area_km2 > 0;

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 mb-3 shadow-lg">
      
      {/* Header Banner */}
      <div className={`flex items-center gap-2 font-bold text-xs mb-4 border-b border-slate-800 pb-3 ${isSpill ? 'text-rose-400' : 'text-emerald-400'}`}>
        {isSpill ? <AlertTriangle className="w-4 h-4 animate-pulse" /> : <CheckCircle className="w-4 h-4" />}
        <span className="tracking-wider">{classification.class_label}</span>
      </div>

      {/* Geometric Properties Grid */}
      <div className="grid grid-cols-2 gap-3 text-xs mb-4">
        
        <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
          <div className="text-slate-400 flex items-center gap-1.5 mb-1.5 text-[10px] uppercase font-semibold">
            <Layers className="w-3.5 h-3.5 text-cyan-400" /> Surface Area
          </div>
          <p className="text-sm font-black text-cyan-300">
            {geometric_properties.area_km2} <span className="text-[10px] font-normal text-slate-400">km²</span>
          </p>
        </div>

        <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
          <div className="text-slate-400 flex items-center gap-1.5 mb-1.5 text-[10px] uppercase font-semibold">
            <Clock className="w-3.5 h-3.5 text-amber-400" /> Estimated Age
          </div>
          <p className="text-sm font-black text-amber-300">
            {isSpill ? `~${geometric_properties.estimated_age_hours}` : '0'} <span className="text-[10px] font-normal text-slate-400">hrs</span>
          </p>
        </div>

        <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
          <div className="text-slate-400 flex items-center gap-1.5 mb-1.5 text-[10px] uppercase font-semibold">
            <Maximize className="w-3.5 h-3.5 text-indigo-400" /> Dimensions (L × W)
          </div>
          <p className="text-sm font-bold text-slate-200">
            {geometric_properties.length_km} <span className="text-slate-500 font-normal text-[10px]">×</span> {geometric_properties.width_km} <span className="text-[10px] font-normal text-slate-400">km</span>
          </p>
        </div>

        <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
          <div className="text-slate-400 flex items-center gap-1.5 mb-1.5 text-[10px] uppercase font-semibold">
            <Activity className="w-3.5 h-3.5 text-rose-400" /> Form Factor (Complexity)
          </div>
          <p className="text-sm font-bold text-slate-200">
            {geometric_properties.form_factor || "0.00"}
          </p>
        </div>

      </div>

      {/* Imagery Row required by PS ID 26143 */}
      {artifacts?.original_sar && (
        <div className="grid grid-cols-2 gap-3 pt-3 border-t border-slate-800">
          <div className="flex flex-col">
            <span className="text-[9px] text-slate-400 mb-1.5 text-center font-bold tracking-widest uppercase">Sentinel-1 Original</span>
            <img src={artifacts.original_sar} alt="Raw SAR" className="w-full h-28 object-cover rounded-lg border border-slate-700 shadow-md" />
          </div>
          <div className="flex flex-col">
            <span className="text-[9px] text-slate-400 mb-1.5 text-center font-bold tracking-widest uppercase">AI Annotation</span>
            <img src={artifacts.processed_sar} alt="Processed SAR" className="w-full h-28 object-cover rounded-lg border border-slate-700 shadow-md" />
          </div>
        </div>
      )}
    </div>
  );
}