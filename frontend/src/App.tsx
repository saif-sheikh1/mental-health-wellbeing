import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Header } from './components/layout/Header';
import { Sidebar } from './components/layout/Sidebar';
import { MentalStateCard } from './components/dashboard/MentalStateCard';
import { RecommendationsCard } from './components/dashboard/RecommendationsCard';
import { SensorOverview } from './components/sensors/SensorOverview';
import { CameraFeed } from './components/facial/CameraFeed';
import { EmotionSelector } from './components/facial/EmotionSelector';
import { EEGChannelsGrid } from './components/eeg/EEGChannelsGrid';
import { PredictionHistory } from './components/history/PredictionHistory';
import { ClinicalReportModal } from './components/history/ClinicalReportModal';
import { LoginScreen } from './components/auth/LoginScreen';
import { IntegratedMasterTab } from './components/dashboard/IntegratedMasterTab';
import { useSensorsStream } from './hooks/useSensorsStream';
import { useDashboardStore } from './store/useDashboardStore';

export function App() {
  // Initialize stream listener for keyboard shortcuts (0-5 & H,S,A,N,F,W,D) & telemetry
  useSensorsStream();

  const { isAuthenticated, activeTab, isDarkMode } = useDashboardStore();

  // Show Login Gate if not authenticated
  if (!isAuthenticated) {
    return <LoginScreen />;
  }

  return (
    <div className={`min-h-screen transition-colors duration-300 ${
      isDarkMode 
        ? 'bg-slate-950 text-slate-100' 
        : 'bg-slate-100 text-slate-900'
    }`}>
      {/* Background Ambient Glows */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-40 -left-40 w-96 h-96 rounded-full bg-cyan-500/10 blur-3xl" />
        <div className="absolute top-1/3 -right-40 w-96 h-96 rounded-full bg-teal-500/10 blur-3xl" />
        <div className="absolute -bottom-40 left-1/3 w-96 h-96 rounded-full bg-emerald-500/10 blur-3xl" />
      </div>

      {/* Top Header */}
      <Header />

      <div className="max-w-[1700px] mx-auto flex flex-col md:flex-row min-h-[calc(100vh-4.5rem)] relative z-10">
        {/* Navigation Sidebar */}
        <Sidebar />

        {/* Main Content Viewport */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8 space-y-6 overflow-x-hidden">
          
          {/* Dynamic Tab Views */}
          <AnimatePresence mode="wait">
            {activeTab === 'master' && <IntegratedMasterTab key="master" />}

            {activeTab === 'dashboard' && (
              <motion.div
                key="dashboard"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.25 }}
                className="space-y-6"
              >
                <MentalStateCard />
                <SensorOverview />
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                  <div className="lg:col-span-6">
                    <CameraFeed />
                  </div>
                  <div className="lg:col-span-6">
                    <EmotionSelector />
                  </div>
                </div>
                <RecommendationsCard />
              </motion.div>
            )}

            {activeTab === 'eeg' && (
              <motion.div
                key="eeg"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.25 }}
                className="space-y-6"
              >
                <EEGChannelsGrid />
                <SensorOverview />
              </motion.div>
            )}

            {activeTab === 'sensors' && (
              <motion.div
                key="sensors"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.25 }}
                className="space-y-6"
              >
                <SensorOverview />
                <EEGChannelsGrid />
              </motion.div>
            )}

            {activeTab === 'facial' && (
              <motion.div
                key="facial"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.25 }}
                className="grid grid-cols-1 lg:grid-cols-12 gap-6"
              >
                <div className="lg:col-span-6">
                  <CameraFeed />
                </div>
                <div className="lg:col-span-6">
                  <EmotionSelector />
                </div>
              </motion.div>
            )}

            {activeTab === 'history' && (
              <motion.div
                key="history"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.25 }}
                className="space-y-6"
              >
                <PredictionHistory />
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>

      {/* Printable Clinical Report Modal */}
      <ClinicalReportModal />
    </div>
  );
}

export default App;
