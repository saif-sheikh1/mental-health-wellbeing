import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  BrainCircuit, 
  Clock, 
  Moon, 
  Sun, 
  Wifi, 
  FileText,
  UserCheck,
  LogOut
} from 'lucide-react';
import { useDashboardStore } from '../../store/useDashboardStore';

export const Header: React.FC = () => {
  const { 
    isDarkMode, 
    setDarkMode, 
    setReportModalOpen,
    patientInfo,
    logout
  } = useDashboardStore();

  const [timeStr, setTimeStr] = useState<string>('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      setTimeStr(now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className={`sticky top-0 z-40 w-full backdrop-blur-xl transition-colors duration-300 border-b ${
      isDarkMode 
        ? 'bg-slate-950/80 border-slate-800/70 text-slate-100' 
        : 'bg-white/80 border-slate-200 text-slate-800'
    }`}>
      <div className="max-w-[1700px] mx-auto px-4 sm:px-6 lg:px-8 h-18 flex items-center justify-between gap-4">
        
        {/* Brand & Logo */}
        <div className="flex items-center gap-3">
          <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-600 via-teal-500 to-emerald-500 shadow-lg shadow-cyan-500/20">
            <BrainCircuit className="w-6 h-6 text-white animate-pulse" />
            <span className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-emerald-400 border-2 border-slate-950 animate-ping" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-cyan-400 via-teal-300 to-emerald-400 bg-clip-text text-transparent">
                MindSense AI
              </h1>
              <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                Live Suite
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              Multimodal Mental Health Diagnostics & Biometric Telemetry Platform
            </p>
          </div>
        </div>

        {/* Center: Live Telemetry Status & Time */}
        <div className="hidden lg:flex items-center gap-3">
          {/* Always Positive Connection Pill */}
          <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full border text-xs font-medium bg-emerald-500/10 border-emerald-500/30 text-emerald-400">
            <Wifi className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
            <span>Live Telemetry Stream Active (256 Hz)</span>
          </div>

          {/* Clock */}
          <div className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-full border text-xs font-mono ${
            isDarkMode ? 'bg-slate-900/60 border-slate-800 text-slate-300' : 'bg-slate-100 border-slate-200 text-slate-700'
          }`}>
            <Clock className="w-3.5 h-3.5 text-cyan-400" />
            <span>{timeStr}</span>
          </div>
        </div>

        {/* Right Actions */}
        <div className="flex items-center gap-2.5">
          {/* Logged in Patient Badge */}
          <div className={`hidden md:flex items-center gap-2 px-3 py-1.5 rounded-xl border text-xs ${
            isDarkMode ? 'bg-slate-900 border-slate-800' : 'bg-slate-100 border-slate-200'
          }`}>
            <UserCheck className="w-4 h-4 text-emerald-400" />
            <div>
              <span className="font-semibold block leading-tight">{patientInfo.name}</span>
              <span className="text-slate-400 text-[10px] block leading-tight">{patientInfo.email}</span>
            </div>
          </div>

          {/* Clinical Report Export Button */}
          <motion.button
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            onClick={() => setReportModalOpen(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 text-xs font-medium transition-all shadow-sm"
          >
            <FileText className="w-4 h-4" />
            <span className="hidden sm:inline">Clinical Report</span>
          </motion.button>

          {/* Dark/Light Mode Toggle */}
          <motion.button
            whileHover={{ scale: 1.08 }}
            whileTap={{ scale: 0.92 }}
            onClick={() => setDarkMode(!isDarkMode)}
            className={`p-2 rounded-xl border transition-all ${
              isDarkMode 
                ? 'bg-slate-900 border-slate-800 text-amber-300 hover:bg-slate-800' 
                : 'bg-slate-100 border-slate-300 text-slate-700 hover:bg-slate-200'
            }`}
            aria-label="Toggle Theme"
          >
            {isDarkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </motion.button>

          {/* Logout / Switch Patient Button */}
          <motion.button
            whileHover={{ scale: 1.04 }}
            whileTap={{ scale: 0.96 }}
            onClick={logout}
            className="p-2 rounded-xl border border-slate-800 bg-slate-900 text-slate-400 hover:text-red-400 hover:border-red-500/30 transition-all"
            title="Sign Out / Switch Patient"
          >
            <LogOut className="w-4 h-4" />
          </motion.button>
        </div>

      </div>
    </header>
  );
};
