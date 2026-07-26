import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { Activity, Brain, Radio, Sparkles } from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip } from 'recharts';
import { useDashboardStore } from '../../store/useDashboardStore';

export const EEGChannelsGrid: React.FC = () => {
  const { readings, isDarkMode } = useDashboardStore();
  const eeg = readings.eeg;

  const channels = [
    {
      id: 'delta',
      name: 'Delta',
      freq: '0.5 – 4 Hz',
      val: eeg.delta,
      desc: 'Deep NREM Sleep & Hypoarousal',
      color: '#3b82f6', // blue
      bg: 'bg-blue-500/10 border-blue-500/30 text-blue-400',
    },
    {
      name: 'Theta',
      id: 'theta',
      freq: '4 – 8 Hz',
      val: eeg.theta,
      desc: 'Deep Relaxation & Intuitive Focus',
      color: '#8b5cf6', // purple
      bg: 'bg-purple-500/10 border-purple-500/30 text-purple-400',
    },
    {
      name: 'Low Alpha',
      id: 'lowAlpha',
      freq: '8 – 10 Hz',
      val: eeg.lowAlpha,
      desc: 'Reflective State & Sensory Resting',
      color: '#06b6d4', // cyan
      bg: 'bg-cyan-500/10 border-cyan-500/30 text-cyan-400',
    },
    {
      name: 'High Alpha',
      id: 'highAlpha',
      freq: '10 – 12 Hz',
      val: eeg.highAlpha,
      desc: 'Peak Calibrated Alpha Coherence',
      color: '#10b981', // emerald
      bg: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400',
    },
    {
      name: 'Low Beta',
      id: 'lowBeta',
      freq: '12 – 15 Hz',
      val: eeg.lowBeta,
      desc: 'Active Alertness & Problem Solving',
      color: '#f59e0b', // amber
      bg: 'bg-amber-500/10 border-amber-500/30 text-amber-400',
    },
    {
      name: 'High Beta',
      id: 'highBeta',
      freq: '15 – 30 Hz',
      val: eeg.highBeta,
      desc: 'Agitation & High Mental Activity',
      color: '#f97316', // orange
      bg: 'bg-orange-500/10 border-orange-500/30 text-orange-400',
    },
    {
      name: 'Low Gamma',
      id: 'lowGamma',
      freq: '30 – 45 Hz',
      val: eeg.lowGamma,
      desc: 'Complex Information Processing',
      color: '#ec4899', // pink
      bg: 'bg-pink-500/10 border-pink-500/30 text-pink-400',
    },
    {
      name: 'Mid Gamma',
      id: 'midGamma',
      freq: '45 – 60 Hz',
      val: eeg.midGamma,
      desc: 'Hyper-vigilance & Cognitive Binding',
      color: '#ef4444', // red
      bg: 'bg-red-500/10 border-red-500/30 text-red-400',
    },
  ];

  // Helper to generate live sparkline data array for each EEG channel
  const makeWaveData = (val: number) => {
    return Array.from({ length: 15 }).map((_, i) => ({
      t: i,
      power: Math.max(0, val + (Math.sin(i * 0.9) * 0.08) + ((Math.random() - 0.5) * 0.04))
    }));
  };

  return (
    <div className={`p-6 rounded-3xl border transition-all duration-300 ${
      isDarkMode 
        ? 'bg-slate-900/80 border-slate-800/80 shadow-xl' 
        : 'bg-white/90 border-slate-200 shadow-md'
    }`}>
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-2xl bg-gradient-to-tr from-cyan-600 to-teal-500 text-white shadow-md shadow-cyan-500/20">
            <Brain className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <h3 className="text-lg font-bold tracking-tight">8-Channel EEG Spectral Decomposition</h3>
            <p className="text-xs text-slate-400">Continuous Electroencephalogram Frequency Band Microvolt Telemetry</p>
          </div>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs text-slate-400">
          <Radio className="w-4 h-4 text-cyan-400 animate-ping" />
          <span>8 Active Leads (Dry Sensors)</span>
        </div>
      </div>

      {/* 8 Channel Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {channels.map((ch) => {
          const waveData = makeWaveData(ch.val);
          const percent = Math.min(100, Math.max(0, Math.round(ch.val * 100)));

          return (
            <motion.div
              key={ch.id}
              whileHover={{ y: -3, scale: 1.01 }}
              className={`p-4 rounded-2xl border transition-all relative overflow-hidden ${
                isDarkMode 
                  ? 'bg-slate-950/60 border-slate-800/70 hover:border-slate-700' 
                  : 'bg-slate-50 border-slate-200 hover:bg-slate-100/80'
              }`}
            >
              {/* Top Header */}
              <div className="flex items-center justify-between mb-2">
                <div>
                  <span className="font-bold text-xs tracking-wide" style={{ color: isDarkMode ? '#f1f5f9' : '#0f172a' }}>
                    {ch.name}
                  </span>
                  <span className="text-[10px] font-mono text-slate-400 ml-1">
                    ({ch.freq})
                  </span>
                </div>
                <span className={`px-2 py-0.5 rounded-full text-[10px] font-mono border ${ch.bg}`}>
                  {ch.val > 0.65 ? 'Dominant' : ch.val > 0.35 ? 'Moderate' : 'Resting'}
                </span>
              </div>

              {/* Value and Progress Bar */}
              <div className="my-2 space-y-1">
                <div className="flex items-baseline justify-between">
                  <span className="font-mono text-xl font-extrabold" style={{ color: ch.color }}>
                    {ch.val.toFixed(3)}
                  </span>
                  <span className="font-mono text-[11px] text-slate-400">
                    {percent}% Power
                  </span>
                </div>

                {/* Animated Progress Bar */}
                <div className="h-2 w-full bg-slate-800/60 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full rounded-full"
                    style={{ backgroundColor: ch.color }}
                    initial={{ width: 0 }}
                    animate={{ width: `${percent}%` }}
                    transition={{ duration: 0.5, ease: 'easeOut' }}
                  />
                </div>
              </div>

              <p className="text-[10px] text-slate-400 line-clamp-1 mb-2">
                {ch.desc}
              </p>

              {/* Animated Recharts Wave Graph for each Channel */}
              <div className="h-16 w-full mt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={waveData} margin={{ top: 2, right: 2, left: -20, bottom: 0 }}>
                    <defs>
                      <linearGradient id={`grad-${ch.id}`} x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={ch.color} stopOpacity={0.4} />
                        <stop offset="95%" stopColor={ch.color} stopOpacity={0.0} />
                      </linearGradient>
                    </defs>
                    <Area
                      type="monotone"
                      dataKey="power"
                      stroke={ch.color}
                      strokeWidth={2}
                      fillOpacity={1}
                      fill={`url(#grad-${ch.id})`}
                      isAnimationActive={true}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
};
