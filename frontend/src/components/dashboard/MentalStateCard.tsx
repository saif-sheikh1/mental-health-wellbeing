import React from 'react';
import { motion } from 'framer-motion';
import { Activity, Brain, Shield, HeartPulse, Sparkles } from 'lucide-react';
import { useDashboardStore } from '../../store/useDashboardStore';
import { getMentalStateMeta } from '../../utils/ranges';

export const MentalStateCard: React.FC = () => {
  const { mentalState, confidence, wellnessScore, isDarkMode } = useDashboardStore();
  const meta = getMentalStateMeta(mentalState);

  // SVG Circular progress math
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (wellnessScore / 100) * circumference;

  return (
    <div className={`relative p-6 rounded-3xl border overflow-hidden transition-all duration-300 ${
      isDarkMode 
        ? 'bg-slate-900/90 border-slate-800/80 shadow-2xl' 
        : 'bg-white/95 border-slate-200 shadow-lg'
    }`}>
      {/* Glow Ambient Accent */}
      <div 
        className="absolute -top-24 -right-24 w-64 h-64 rounded-full blur-3xl opacity-20 pointer-events-none transition-all duration-500"
        style={{ backgroundColor: meta.color }}
      />

      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 relative z-10">
        
        {/* Left Info */}
        <div className="space-y-3 flex-1">
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-1 rounded-full text-xs font-mono font-semibold uppercase tracking-wider border bg-slate-800/80 text-cyan-400 border-cyan-500/30">
              Primary Diagnostic Output
            </span>
            <span className={`px-2.5 py-1 rounded-full text-xs font-mono font-bold border ${meta.badgeBg} ${meta.badgeText}`}>
              {mentalState.replace('_', ' ')}
            </span>
          </div>

          <div>
            <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight" style={{ color: isDarkMode ? '#f8fafc' : '#0f172a' }}>
              {meta.label}
            </h2>
            <p className="text-xs sm:text-sm text-slate-400 mt-1 max-w-xl leading-relaxed">
              {meta.description}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-4 pt-2">
            {/* Confidence Score Pill */}
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-950/60 border border-slate-800 text-xs">
              <Brain className="w-4 h-4 text-cyan-400" />
              <span className="text-slate-400">AI Confidence:</span>
              <span className="font-mono font-bold text-cyan-300">{confidence}%</span>
            </div>

            {/* Status Signal Pill */}
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-950/60 border border-slate-800 text-xs">
              <Activity className="w-4 h-4 text-emerald-400 animate-pulse" />
              <span className="text-slate-400">Signal Stability:</span>
              <span className="font-mono font-bold text-emerald-300">Optimal (256Hz)</span>
            </div>
          </div>
        </div>

        {/* Right Wellness Meter (Circular SVG Ring) */}
        <div className="flex items-center gap-4 self-center md:self-auto p-4 rounded-2xl bg-slate-950/60 border border-slate-800/80">
          <div className="relative w-28 h-28 flex items-center justify-center">
            <svg className="w-full h-full transform -rotate-90">
              {/* Track */}
              <circle
                cx="56"
                cy="56"
                r={radius}
                className="text-slate-800"
                strokeWidth="8"
                stroke="currentColor"
                fill="transparent"
              />
              {/* Indicator */}
              <motion.circle
                cx="56"
                cy="56"
                r={radius}
                stroke={meta.color}
                strokeWidth="8"
                strokeDasharray={circumference}
                initial={{ strokeDashoffset: circumference }}
                animate={{ strokeDashoffset }}
                transition={{ duration: 1, ease: 'easeOut' }}
                strokeLinecap="round"
                fill="transparent"
              />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
              <span className="font-mono text-2xl font-black tracking-tight" style={{ color: meta.color }}>
                {wellnessScore}
              </span>
              <span className="text-[10px] font-mono uppercase text-slate-400 tracking-wider">
                Wellness
              </span>
            </div>
          </div>

          <div className="space-y-1">
            <div className="flex items-center gap-1 text-xs font-semibold text-slate-200">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" /> Index Metric
            </div>
            <p className="text-[11px] text-slate-400 max-w-[120px] leading-tight">
              Integrated score based on GSR, PPG, & 8 EEG bands.
            </p>
          </div>
        </div>

      </div>
    </div>
  );
};
