import React, { useState, useEffect } from 'react';
import { Wind, Waves, MapPin, Clock, Play, Pause } from 'lucide-react';

export default function MetoceanSimulator({ hindcastData, activeStep, setActiveStep }) {
  const [isPlaying, setIsPlaying] = useState(false);

  // Auto-Play Animation Engine
  useEffect(() => {
    let interval;
    if (isPlaying && hindcastData?.hindcast_trajectory) {
      interval = setInterval(() => {
        setActiveStep((prevStep) => {
          if (prevStep >= hindcastData.hindcast_trajectory.length - 1) {
            setIsPlaying(false);
            return prevStep;
          }
          return prevStep + 1;
        });
      }, 250); // 250ms for a smoother, visible animation
    }
    return () => clearInterval(interval);
  }, [isPlaying, hindcastData, setActiveStep]);

  if (!hindcastData || !hindcastData.origin_estimate) {
    return (
      <div className="flex items-center justify-center h-48 text-slate-500 text-xs font-mono border border-slate-800 border-dashed rounded-xl mx-2">
        NO METOCEAN DATA. RUN PIPELINE.
      </div>
    );
  }

  const { forcing_parameters, origin_estimate } = hindcastData;

  // Safe date parsing
  const originDate = new Date(origin_estimate.timestamp);
  const formattedDate = isNaN(originDate) ? "Calculating..." : originDate.toLocaleString();

  // Protect against undefined steps
  const currentStepData = hindcastData.hindcast_trajectory[activeStep || 0];

  const handlePlayToggle = () => {
    if (!isPlaying && activeStep >= hindcastData.hindcast_trajectory.length - 1) {
      setActiveStep(0);
    }
    setIsPlaying(!isPlaying);
  };

  return (
    <div className="space-y-4 pb-4">
      {/* Forcing Parameters Card */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-lg relative overflow-hidden">
        <div className="absolute top-0 right-0 p-2 opacity-10">
          <Wind className="w-16 h-16" />
        </div>
        <h3 className="text-[10px] uppercase font-bold text-slate-400 mb-3 tracking-widest border-b border-slate-800 pb-2">
          Environmental Forcing
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="flex items-center gap-1.5 text-cyan-400 mb-1.5">
              <Wind className="w-4 h-4" /> <span className="text-xs font-semibold">Surface Wind</span>
            </div>
            <p className="text-sm font-mono text-slate-200">
              {forcing_parameters?.wind_speed_ms || '0.0'} m/s
            </p>
          </div>
          <div>
            <div className="flex items-center gap-1.5 text-blue-400 mb-1.5">
              <Waves className="w-4 h-4" /> <span className="text-xs font-semibold">Ocean Current</span>
            </div>
            <p className="text-sm font-mono text-slate-200">
              {forcing_parameters?.current_speed_ms || '0.0'} m/s
            </p>
          </div>
        </div>
        <p className="text-[9px] text-slate-500 mt-4 text-right font-mono uppercase tracking-widest">
          SRC: {forcing_parameters?.source || 'Unknown'}
        </p>
      </div>

      {/* Origin Estimate Card */}
      <div className="bg-slate-900/90 border border-rose-900/30 rounded-xl p-4 shadow-lg relative overflow-hidden">
        <h3 className="text-[10px] uppercase font-bold text-rose-400 mb-3 tracking-widest border-b border-slate-800 pb-2 flex justify-between">
          <span>Estimated Origin Point</span>
          <span>-24 Hrs</span>
        </h3>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between bg-slate-950/50 p-2.5 rounded border border-slate-800/80">
            <div className="flex items-center gap-2 text-slate-400">
              <MapPin className="w-4 h-4 text-rose-400" /> <span className="text-xs font-semibold">GPS</span>
            </div>
            <p className="text-xs font-mono text-slate-200">
              {origin_estimate.latitude}, {origin_estimate.longitude}
            </p>
          </div>
          
          <div className="flex items-center justify-between bg-slate-950/50 p-2.5 rounded border border-slate-800/80">
            <div className="flex items-center gap-2 text-slate-400">
              <Clock className="w-4 h-4 text-amber-400" /> <span className="text-xs font-semibold">Time</span>
            </div>
            <p className="text-[10px] font-mono text-slate-300">
              {formattedDate}
            </p>
          </div>
        </div>
      </div>

      {/* Time Scrubber (Animation Engine) */}
      {hindcastData.hindcast_trajectory && (
        <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-lg">
          <div className="flex justify-between items-center mb-4">
            <span className="text-xs font-bold text-slate-400">
              T{currentStepData?.step_hour} Hrs
            </span>
            <span className="text-[10px] text-cyan-400 font-mono">
              {currentStepData?.timestamp ? new Date(currentStepData.timestamp).toLocaleTimeString() : ''}
            </span>
          </div>
          
          <div className="flex items-center gap-3">
            <button 
              onClick={handlePlayToggle}
              className="bg-blue-600 hover:bg-blue-500 text-white p-2 rounded-full shadow-lg transition-colors focus:outline-none flex-shrink-0"
            >
              {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current pl-0.5" />}
            </button>
            
            <input 
              type="range" 
              min="0" 
              max={hindcastData.hindcast_trajectory.length - 1} 
              value={activeStep || 0} 
              onChange={(e) => {
                setIsPlaying(false); // Pause if user manually drags
                setActiveStep(parseInt(e.target.value));
              }}
              className="w-full h-1.5 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500 focus:outline-none"
            />
          </div>
          
          <div className="flex justify-between text-[9px] text-slate-500 mt-3 font-mono uppercase tracking-widest pl-10">
            <span>Detection (0 Hrs)</span>
            <span>Origin (-24 Hrs)</span>
          </div>
        </div>
      )}
    </div>
  );
}