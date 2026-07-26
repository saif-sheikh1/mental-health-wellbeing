import React from 'react';
import { motion } from 'framer-motion';
import { 
  Activity, 
  Brain, 
  Camera, 
  History, 
  LayoutDashboard, 
  Sparkles, 
  Zap 
} from 'lucide-react';
import { useDashboardStore } from '../../store/useDashboardStore';
import { ActiveTab } from '../../types';

export const Sidebar: React.FC = () => {
  const { activeTab, setActiveTab, isDarkMode } = useDashboardStore();

  const navItems: { id: ActiveTab; label: string; icon: React.FC<{ className?: string }> }[] = [
    { id: 'master', label: 'Live Master Suite', icon: Sparkles },
    { id: 'dashboard', label: 'Main Overview', icon: LayoutDashboard },
    { id: 'eeg', label: 'EEG 8-Channel', icon: Brain },
    { id: 'sensors', label: 'Sensor Array', icon: Activity },
    { id: 'facial', label: 'Facial Vision', icon: Camera },
    { id: 'history', label: 'Logs & Reports', icon: History },
  ];

  return (
    <aside className={`w-full md:w-64 shrink-0 transition-colors duration-300 border-r ${
      isDarkMode 
        ? 'bg-slate-950/70 border-slate-800/70 text-slate-300' 
        : 'bg-slate-50/90 border-slate-200 text-slate-700'
    }`}>
      <div className="p-4 space-y-6">
        
        {/* Navigation Section */}
        <div>
          <h2 className="text-[11px] font-mono uppercase tracking-wider text-slate-500 mb-3 px-3">
            Diagnostic Workspace
          </h2>
          <nav className="space-y-1.5">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;

              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`relative w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl font-medium text-sm transition-all duration-200 ${
                    isActive
                      ? isDarkMode
                        ? 'text-cyan-300 font-semibold'
                        : 'text-cyan-700 font-semibold'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                  }`}
                >
                  {isActive && (
                    <motion.div
                      layoutId="activeTabBackground"
                      className={`absolute inset-0 rounded-xl ${
                        isDarkMode
                          ? 'bg-gradient-to-r from-cyan-500/20 to-teal-500/10 border border-cyan-500/30'
                          : 'bg-cyan-50 border border-cyan-200'
                      }`}
                      transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                    />
                  )}
                  <Icon className={`w-4 h-4 z-10 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                  <span className="z-10">{item.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* System Specs Box */}
        <div className={`p-3.5 rounded-2xl border text-xs space-y-2 ${
          isDarkMode ? 'bg-slate-900/60 border-slate-800/80 text-slate-400' : 'bg-white border-slate-200 text-slate-600'
        }`}>
          <div className="flex items-center justify-between font-mono text-[11px] text-cyan-400">
            <span className="flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5" /> Model Engine
            </span>
            <span>BiLSTM + ViT</span>
          </div>
          <p className="text-[11px] leading-relaxed text-slate-400">
            Multimodal BiLSTM Attention Sensor RNN & Residual Vision Transformer.
          </p>
          <div className="pt-1 flex items-center justify-between text-[10px] font-mono border-t border-slate-800/50 text-slate-500">
            <span>Sampling: 256 Hz</span>
            <span>Latency: &lt;12ms</span>
          </div>
        </div>

      </div>
    </aside>
  );
};
