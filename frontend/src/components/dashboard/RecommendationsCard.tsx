import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  AlertCircle, 
  CheckCircle2, 
  HeartHandshake, 
  Stethoscope, 
  Lightbulb, 
  ShieldAlert, 
  Wind, 
  Sparkles,
  CheckSquare,
  Square
} from 'lucide-react';
import { useDashboardStore } from '../../store/useDashboardStore';

export const RecommendationsCard: React.FC = () => {
  const { recommendations, mentalState, isDarkMode } = useDashboardStore();
  const [activeTab, setActiveTab] = useState<'first_aid' | 'lifestyle' | 'professional' | 'breathing'>('first_aid');
  const [completedItems, setCompletedItems] = useState<Record<string, boolean>>({});

  const recs = recommendations || {
    trend: 'stable',
    trend_message: 'Physiological telemetry is stable across electrodermal and spectral bands.',
    first_aid: ['Box Breathing 4s Inhale, 4s Hold, 4s Exhale, 4s Hold', 'Hydrate with 250ml electrolyte water'],
    lifestyle: ['7-8 hours sleep hygiene', '20min outdoor brisk walk'],
    professional: ['Biometric telemetry report ready for export'],
    emotion_tip: 'Positive affect detected.'
  };

  const toggleCheck = (item: string) => {
    setCompletedItems((prev) => ({ ...prev, [item]: !prev[item] }));
  };

  const tabs: { id: 'first_aid' | 'lifestyle' | 'professional' | 'breathing'; label: string; icon: React.FC<{ className?: string }>; color: string }[] = [
    { id: 'first_aid', label: 'First Aid (Immediate)', icon: AlertCircle, color: 'text-red-400' },
    { id: 'lifestyle', label: 'Lifestyle Protocol', icon: HeartHandshake, color: 'text-emerald-400' },
    { id: 'professional', label: 'Clinical Care', icon: Stethoscope, color: 'text-cyan-400' },
    { id: 'breathing', label: 'Box Breathing Guide', icon: Wind, color: 'text-teal-400' },
  ];

  const getActiveList = () => {
    if (activeTab === 'first_aid') return recs.first_aid || [];
    if (activeTab === 'lifestyle') return recs.lifestyle || [];
    return recs.professional || [];
  };

  return (
    <div className={`p-6 rounded-3xl border transition-all duration-300 ${
      isDarkMode 
        ? 'bg-slate-900/80 border-slate-800/80 shadow-xl' 
        : 'bg-white/90 border-slate-200 shadow-md'
    }`}>
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-2xl bg-gradient-to-tr from-cyan-600 to-teal-500 text-white shadow-md shadow-cyan-500/20">
            <Sparkles className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <h3 className="text-base font-bold tracking-tight">Enhanced AI Clinical Recommendations & Interventions</h3>
            <p className="text-xs text-slate-400">Automated copings, vagal regulation, & habit protocols for {mentalState}</p>
          </div>
        </div>

        {/* Trend Urgency Pill */}
        <div className={`px-3.5 py-1.5 rounded-full text-xs font-mono font-semibold border flex items-center gap-2 ${
          recs.trend === 'worsening'
            ? 'bg-red-500/15 border-red-500/40 text-red-400 animate-pulse'
            : recs.trend === 'improving'
              ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-400'
              : 'bg-cyan-500/15 border-cyan-500/40 text-cyan-400'
        }`}>
          <ShieldAlert className="w-4 h-4" />
          <span className="capitalize">Physiological Trend: {recs.trend}</span>
        </div>
      </div>

      {/* Emotion Tip Callout */}
      {recs.emotion_tip && (
        <div className="mb-4 p-3.5 rounded-2xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs flex items-center gap-2.5">
          <Lightbulb className="w-4 h-4 text-cyan-400 shrink-0" />
          <span className="leading-relaxed">{recs.emotion_tip}</span>
        </div>
      )}

      {/* Tabs Bar */}
      <div className="flex border-b border-slate-800/60 mb-4 gap-2 overflow-x-auto pb-1 sm:pb-0">
        {tabs.map((t) => {
          const Icon = t.icon;
          const isActive = activeTab === t.id;

          return (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              className={`pb-2.5 px-3 text-xs font-semibold flex items-center gap-1.5 border-b-2 transition-all whitespace-nowrap ${
                isActive
                  ? `border-cyan-400 text-cyan-300 font-bold`
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              }`}
            >
              <Icon className={`w-3.5 h-3.5 ${t.color}`} />
              <span>{t.label}</span>
            </button>
          );
        })}
      </div>

      {/* Tab Content */}
      <AnimatePresence mode="wait">
        {activeTab === 'breathing' ? (
          <motion.div
            key="breathing"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            className="p-6 rounded-2xl bg-slate-950/70 border border-slate-800 text-center flex flex-col items-center justify-center space-y-4"
          >
            <div className="relative w-28 h-28 flex items-center justify-center">
              <div className="absolute inset-0 rounded-full border-2 border-cyan-400/30 animate-ping" />
              <div className="w-20 h-20 rounded-full bg-gradient-to-tr from-cyan-500/30 to-emerald-500/30 border border-cyan-400/60 flex items-center justify-center animate-pulse-ring">
                <Wind className="w-8 h-8 text-cyan-300" />
              </div>
            </div>
            <div>
              <h4 className="text-sm font-bold text-slate-200">Interactive 4-4-4-4 Box Breathing Guide</h4>
              <p className="text-xs text-slate-400 mt-1 max-w-md">
                Inhale 4s → Hold 4s → Exhale 4s → Hold 4s. Repeat 4 times to stimulate the parasympathetic nervous system and reduce electrodermal arousal.
              </p>
            </div>
          </motion.div>
        ) : (
          <motion.ul
            key={activeTab}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.2 }}
            className="space-y-2.5"
          >
            {getActiveList().map((item, idx) => {
              const isChecked = !!completedItems[item];

              return (
                <li 
                  key={idx} 
                  onClick={() => toggleCheck(item)}
                  className={`flex items-start gap-3 p-3.5 rounded-2xl border text-xs leading-relaxed cursor-pointer transition-all ${
                    isChecked
                      ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300 line-through opacity-75'
                      : isDarkMode 
                        ? 'bg-slate-950/50 border-slate-800/60 text-slate-300 hover:border-slate-700' 
                        : 'bg-slate-50 border-slate-200 text-slate-700 hover:bg-slate-100'
                  }`}
                >
                  {isChecked ? (
                    <CheckSquare className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  ) : (
                    <Square className="w-4 h-4 text-slate-500 shrink-0 mt-0.5" />
                  )}
                  <span>{item}</span>
                </li>
              );
            })}
          </motion.ul>
        )}
      </AnimatePresence>
    </div>
  );
};
