import React from 'react';
import { motion } from 'framer-motion';
import { 
  AlertOctagon, 
  Activity, 
  Brain, 
  CheckCircle2, 
  Flame, 
  Moon, 
  Zap 
} from 'lucide-react';
import { useDashboardStore } from '../../store/useDashboardStore';
import { ShortcutMode } from '../../types';

export const StateShortcutBar: React.FC = () => {
  const { shortcutMode, setShortcutMode, isDarkMode } = useDashboardStore();

  const shortcutButtons: {
    mode: ShortcutMode;
    key: string;
    label: string;
    description: string;
    icon: React.FC<{ className?: string }>;
    activeBg: string;
    borderColor: string;
    textColor: string;
  }[] = [
    {
      mode: 0,
      key: '0',
      label: 'No Signal',
      description: 'GSR: 0 | PPG: 0 | EEG: 0',
      icon: AlertOctagon,
      activeBg: 'bg-slate-500/20',
      borderColor: 'border-slate-500/50',
      textColor: 'text-slate-400',
    },
    {
      mode: 1,
      key: '1',
      label: 'Normal',
      description: 'GSR: 1200-1800 | PPG: 60-80',
      icon: CheckCircle2,
      activeBg: 'bg-emerald-500/20',
      borderColor: 'border-emerald-500/50',
      textColor: 'text-emerald-400',
    },
    {
      mode: 2,
      key: '2',
      label: 'Mod. Stress',
      description: 'GSR: 2200-2600 | PPG: 95-110',
      icon: Activity,
      activeBg: 'bg-amber-500/20',
      borderColor: 'border-amber-500/50',
      textColor: 'text-amber-400',
    },
    {
      mode: 3,
      key: '3',
      label: 'Low Anxiety',
      description: 'GSR: 2600-3000 | PPG: 110-130',
      icon: Zap,
      activeBg: 'bg-orange-500/20',
      borderColor: 'border-orange-500/50',
      textColor: 'text-orange-400',
    },
    {
      mode: 4,
      key: '4',
      label: 'Panic State',
      description: 'GSR: 3000-4095 | PPG: 130-160',
      icon: Flame,
      activeBg: 'bg-red-500/20',
      borderColor: 'border-red-500/50',
      textColor: 'text-red-400',
    },
    {
      mode: 5,
      key: '5',
      label: 'Low Depression',
      description: 'GSR: 1000-1400 | PPG: 55-70',
      icon: Moon,
      activeBg: 'bg-purple-500/20',
      borderColor: 'border-purple-500/50',
      textColor: 'text-purple-400',
    },
  ];

  return (
    <div className={`p-4 rounded-3xl border transition-all duration-300 ${
      isDarkMode 
        ? 'bg-slate-900/80 border-slate-800/80 shadow-xl' 
        : 'bg-white/90 border-slate-200 shadow-md'
    }`}>
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-3 mb-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
            <h3 className="text-sm font-bold tracking-tight">Mental State Simulator Shortcuts</h3>
          </div>
          <p className="text-xs text-slate-400">
            Press keys <kbd className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-cyan-300 font-mono text-[10px]">0-5</kbd> on your keyboard to instantly trigger realistic sensor range simulations without changing the backend.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
        {shortcutButtons.map((btn) => {
          const Icon = btn.icon;
          const isSelected = shortcutMode === btn.mode;

          return (
            <motion.button
              key={btn.key}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              onClick={() => setShortcutMode(btn.mode)}
              className={`relative flex flex-col p-3 rounded-2xl border text-left transition-all duration-200 ${
                isSelected
                  ? `${btn.activeBg} ${btn.borderColor} shadow-lg shadow-black/20 ring-1 ring-cyan-500/30`
                  : isDarkMode
                    ? 'bg-slate-950/60 border-slate-800/70 hover:border-slate-700 hover:bg-slate-900/60'
                    : 'bg-slate-50 border-slate-200 hover:bg-slate-100'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className={`w-5 h-5 rounded-md flex items-center justify-center font-mono font-bold text-xs ${
                  isSelected ? 'bg-cyan-500 text-slate-950' : 'bg-slate-800 text-slate-400'
                }`}>
                  {btn.key}
                </span>
                <Icon className={`w-4 h-4 ${isSelected ? btn.textColor : 'text-slate-500'}`} />
              </div>
              <span className={`font-bold text-xs ${isSelected ? btn.textColor : isDarkMode ? 'text-slate-200' : 'text-slate-800'}`}>
                {btn.label}
              </span>
              <span className="text-[10px] font-mono text-slate-400 mt-1 line-clamp-1">
                {btn.description}
              </span>
            </motion.button>
          );
        })}
      </div>
    </div>
  );
};
