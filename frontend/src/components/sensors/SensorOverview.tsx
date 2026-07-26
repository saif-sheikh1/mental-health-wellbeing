import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { Activity, Camera, Heart, Zap } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line } from 'recharts';
import { useDashboardStore } from '../../store/useDashboardStore';

export const SensorOverview: React.FC = () => {
  const { readings, isDarkMode } = useDashboardStore();

  // Synthetic historical trend sparkline generator based on current reading
  const generateSparkline = (baseVal: number, range: number) => {
    return Array.from({ length: 12 }).map((_, i) => ({
      i,
      val: Math.max(0, baseVal + (Math.sin(i * 0.8) * range) + ((Math.random() - 0.5) * range * 0.5))
    }));
  };

  const gsrData = useMemo(() => generateSparkline(readings.gsr, readings.gsr * 0.1), [readings.gsr]);
  const ppgData = useMemo(() => generateSparkline(readings.ppg, 8), [readings.ppg]);
  const eegData = useMemo(() => generateSparkline(readings.eeg.highAlpha * 100, 15), [readings.eeg.highAlpha]);
  const facialData = useMemo(() => generateSparkline(readings.facialConfidence, 5), [readings.facialConfidence]);

  const cards = [
    {
      id: 'gsr',
      title: 'GSR Electrodermal',
      sub: 'Skin Conductance (Arousal)',
      val: `${readings.gsr} µS`,
      status: readings.gsr > 2800 ? 'High Arousal' : readings.gsr > 2000 ? 'Elevated' : 'Normal',
      statusColor: readings.gsr > 2800 ? 'text-red-400 bg-red-500/10 border-red-500/30' : 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
      icon: Zap,
      iconColor: 'text-amber-400',
      strokeColor: '#f59e0b',
      sparkline: gsrData,
    },
    {
      id: 'ppg',
      title: 'PPG Heart Rate',
      sub: 'Photoplethysmogram BPM',
      val: `${readings.ppg} BPM`,
      status: readings.ppg > 110 ? 'Tachycardia' : readings.ppg > 85 ? 'Elevated' : 'Optimal',
      statusColor: readings.ppg > 110 ? 'text-orange-400 bg-orange-500/10 border-orange-500/30' : 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30',
      icon: Heart,
      iconColor: 'text-rose-400',
      strokeColor: '#f43f5e',
      sparkline: ppgData,
    },
    {
      id: 'eeg',
      title: 'EEG High Alpha',
      sub: 'Relaxation Coherence (10-12Hz)',
      val: `${readings.eeg.highAlpha.toFixed(2)} µV`,
      status: readings.eeg.highAlpha > 0.60 ? 'High Coherence' : 'Low Coherence',
      statusColor: readings.eeg.highAlpha > 0.60 ? 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30' : 'text-purple-400 bg-purple-500/10 border-purple-500/30',
      icon: Activity,
      iconColor: 'text-cyan-400',
      strokeColor: '#06b6d4',
      sparkline: eegData,
    },
    {
      id: 'facial',
      title: 'Facial Vision Classifier',
      sub: 'Primary Expression Confidence',
      val: `${readings.facial} (${readings.facialConfidence}%)`,
      status: 'Active Feed',
      statusColor: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/30',
      icon: Camera,
      iconColor: 'text-teal-400',
      strokeColor: '#14b8a6',
      sparkline: facialData,
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((c) => {
        const Icon = c.icon;

        return (
          <motion.div
            key={c.id}
            whileHover={{ y: -4, scale: 1.01 }}
            className={`p-5 rounded-3xl border transition-all duration-300 relative overflow-hidden ${
              isDarkMode 
                ? 'bg-slate-900/80 border-slate-800/80 shadow-lg' 
                : 'bg-white/90 border-slate-200 shadow-md'
            }`}
          >
            {/* Top Bar */}
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <div className={`p-2 rounded-xl bg-slate-950/80 border border-slate-800 ${c.iconColor}`}>
                  <Icon className="w-4 h-4" />
                </div>
                <div>
                  <h4 className="text-xs font-bold tracking-tight">{c.title}</h4>
                  <p className="text-[10px] text-slate-400">{c.sub}</p>
                </div>
              </div>
              
              <span className={`px-2 py-0.5 rounded-full text-[10px] font-mono border ${c.statusColor}`}>
                {c.status}
              </span>
            </div>

            {/* Value Display */}
            <div className="my-3 flex items-baseline justify-between">
              <span className="font-mono text-2xl font-black tracking-tight" style={{ color: isDarkMode ? '#f8fafc' : '#0f172a' }}>
                {c.val}
              </span>
              <span className="text-[10px] font-mono text-slate-500">
                Live {readings.lastUpdated}
              </span>
            </div>

            {/* Mini Sparkline Chart */}
            <div className="h-10 w-full mt-2">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={c.sparkline}>
                  <Line
                    type="monotone"
                    dataKey="val"
                    stroke={c.strokeColor}
                    strokeWidth={2}
                    dot={false}
                    isAnimationActive={true}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
};
