import React from 'react';
import { motion } from 'framer-motion';
import { FacialEmotion } from '../../types';
import { useDashboardStore } from '../../store/useDashboardStore';
import { ALL_EMOTIONS } from '../../utils/ranges';

const emotionConfigs: Record<FacialEmotion, { emoji: string; color: string; border: string }> = {
  Happy: { emoji: '😊', color: 'from-amber-400 to-emerald-400', border: 'border-emerald-500/50' },
  Sad: { emoji: '😔', color: 'from-blue-400 to-indigo-500', border: 'border-blue-500/50' },
  Angry: { emoji: '😠', color: 'from-rose-500 to-red-600', border: 'border-rose-500/50' },
  Neutral: { emoji: '😐', color: 'from-slate-400 to-slate-500', border: 'border-slate-500/50' },
  Fear: { emoji: '😨', color: 'from-purple-400 to-violet-600', border: 'border-purple-500/50' },
  Surprise: { emoji: '😲', color: 'from-cyan-400 to-teal-500', border: 'border-cyan-500/50' },
  Disgust: { emoji: '🤢', color: 'from-lime-500 to-emerald-600', border: 'border-lime-500/50' },
};

export const EmotionSelector: React.FC = () => {
  const { activeEmotion, setActiveEmotion, readings, isDarkMode } = useDashboardStore();
  const probs = readings.facialProbabilities || {};

  return (
    <div className={`p-5 rounded-3xl border transition-all duration-300 ${
      isDarkMode 
        ? 'bg-slate-900/80 border-slate-800/80 shadow-xl' 
        : 'bg-white/90 border-slate-200 shadow-md'
    }`}>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-bold tracking-tight">Facial Affect Classification</h3>
          <p className="text-xs text-slate-400">
            Realtime 7-emotion ViT convolutional response probabilities
          </p>
        </div>
        <div className="px-2.5 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 text-xs font-mono">
          Primary: {activeEmotion} ({readings.facialConfidence}%)
        </div>
      </div>

      {/* Emotion Selector Buttons (No visible shortcut labels) */}
      <div className="grid grid-cols-7 gap-2 mb-5">
        {ALL_EMOTIONS.map((emotion) => {
          const cfg = emotionConfigs[emotion];
          const isSelected = activeEmotion === emotion;
          const prob = probs[emotion] || 0;

          return (
            <motion.button
              key={emotion}
              whileHover={{ scale: 1.05, y: -2 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setActiveEmotion(emotion)}
              className={`relative flex flex-col items-center justify-center p-2.5 rounded-2xl border transition-all ${
                isSelected
                  ? `${cfg.border} bg-slate-800/80 shadow-lg ring-2 ring-cyan-500/40`
                  : isDarkMode
                    ? 'bg-slate-950/40 border-slate-800/60 hover:bg-slate-900/60'
                    : 'bg-slate-50 border-slate-200 hover:bg-slate-100'
              }`}
            >
              <span className="text-2xl mb-1">{cfg.emoji}</span>
              <span className={`text-[11px] font-bold ${isSelected ? 'text-cyan-300' : 'text-slate-400'}`}>
                {emotion}
              </span>

              {/* Confidence Badge */}
              <span className={`mt-1 font-mono text-[10px] px-1.5 py-0.5 rounded-full ${
                isSelected ? 'bg-cyan-500/20 text-cyan-300 font-bold' : 'text-slate-500'
              }`}>
                {prob}%
              </span>
            </motion.button>
          );
        })}
      </div>

      {/* Realistic Emotion Probabilities Distribution Breakdown */}
      <div className="space-y-2 pt-2 border-t border-slate-800/60">
        <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-1">
          <span>Realistic Confidence Distribution (Sums ~100%)</span>
          <span className="text-cyan-400">7-Emotion Classifier</span>
        </div>

        <div className="space-y-1.5">
          {ALL_EMOTIONS.map((emotion) => {
            const prob = probs[emotion] || 0;
            const isSelected = activeEmotion === emotion;

            return (
              <div key={emotion} className="flex items-center gap-3 text-xs">
                <span className={`w-16 font-medium text-[11px] ${isSelected ? 'text-cyan-300 font-bold' : 'text-slate-400'}`}>
                  {emotion}
                </span>
                <div className="flex-1 h-2 rounded-full bg-slate-800/60 overflow-hidden">
                  <motion.div
                    className={`h-full rounded-full ${
                      isSelected 
                        ? 'bg-gradient-to-r from-cyan-500 to-emerald-400' 
                        : 'bg-slate-600/50'
                    }`}
                    initial={{ width: 0 }}
                    animate={{ width: `${prob}%` }}
                    transition={{ duration: 0.5, ease: 'easeOut' }}
                  />
                </div>
                <span className={`w-10 font-mono text-right text-[11px] ${isSelected ? 'text-cyan-300 font-bold' : 'text-slate-500'}`}>
                  {prob}%
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
