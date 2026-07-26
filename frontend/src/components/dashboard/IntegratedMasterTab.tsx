import React from 'react';
import { motion } from 'framer-motion';
import { MentalStateCard } from './MentalStateCard';
import { SensorOverview } from '../sensors/SensorOverview';
import { CameraFeed } from '../facial/CameraFeed';
import { EmotionSelector } from '../facial/EmotionSelector';
import { EEGChannelsGrid } from '../eeg/EEGChannelsGrid';
import { RecommendationsCard } from './RecommendationsCard';
import { PredictionHistory } from '../history/PredictionHistory';

export const IntegratedMasterTab: React.FC = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.3 }}
      className="space-y-6"
    >
      {/* 1. Primary Mental State Diagnostic Card & Wellness Gauge */}
      <MentalStateCard />

      {/* 2. Core Sensor Telemetry Grid (GSR, PPG, EEG, Facial) */}
      <SensorOverview />

      {/* 3. Live Vision Camera Feed & Facial Emotion Probabilities Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-6">
          <CameraFeed />
        </div>
        <div className="lg:col-span-6">
          <EmotionSelector />
        </div>
      </div>

      {/* 4. Enhanced AI Recommendations & Interactive Coping Checklist */}
      <RecommendationsCard />

      {/* 5. 8-Channel EEG Spectral Decomposition Monitor */}
      <EEGChannelsGrid />

      {/* 6. Realtime Diagnostic History Log & Exportable Patient Records */}
      <PredictionHistory />
    </motion.div>
  );
};
