import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { History, Search, Filter, Download, ArrowUpDown } from 'lucide-react';
import { useDashboardStore } from '../../store/useDashboardStore';
import { getMentalStateMeta } from '../../utils/ranges';

export const PredictionHistory: React.FC = () => {
  const { history, setReportModalOpen, isDarkMode } = useDashboardStore();
  const [searchTerm, setSearchTerm] = useState('');
  const [filterState, setFilterState] = useState<string>('ALL');

  const filteredHistory = history.filter((item) => {
    const matchesSearch = item.timestamp.includes(searchTerm) || item.state.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterState === 'ALL' || item.state === filterState;
    return matchesSearch && matchesFilter;
  });

  return (
    <div className={`p-6 rounded-3xl border transition-all duration-300 ${
      isDarkMode 
        ? 'bg-slate-900/80 border-slate-800/80 shadow-xl' 
        : 'bg-white/90 border-slate-200 shadow-md'
    }`}>
      {/* Header */}
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-2xl bg-cyan-500/10 text-cyan-400">
            <History className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-lg font-bold tracking-tight">Diagnostic Prediction History & Logs</h3>
            <p className="text-xs text-slate-400">Realtime logged telemetry entries and state assessments</p>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setReportModalOpen(true)}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 text-xs font-semibold transition-all"
          >
            <Download className="w-4 h-4" /> Export Report
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 mb-4">
        {/* Search */}
        <div className="relative w-full sm:w-64">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
          <input
            type="text"
            placeholder="Search by state or time..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className={`w-full pl-9 pr-4 py-2 rounded-xl border text-xs outline-none transition-all ${
              isDarkMode 
                ? 'bg-slate-950/80 border-slate-800 text-slate-200 focus:border-cyan-500' 
                : 'bg-slate-50 border-slate-200 text-slate-800 focus:border-cyan-500'
            }`}
          />
        </div>

        {/* Filter Pills */}
        <div className="flex items-center gap-1.5 overflow-x-auto w-full sm:w-auto pb-1 sm:pb-0">
          {['ALL', 'NORMAL', 'MODERATE_STRESS', 'HIGH_ANXIETY', 'PANIC_STATE', 'DEPRESSION'].map((st) => (
            <button
              key={st}
              onClick={() => setFilterState(st)}
              className={`px-2.5 py-1 rounded-lg text-[11px] font-mono transition-all border ${
                filterState === st
                  ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-300 font-bold'
                  : 'bg-slate-800/40 border-slate-800 text-slate-400 hover:bg-slate-800'
              }`}
            >
              {st}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-2xl border border-slate-800/60">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className={`border-b font-mono ${isDarkMode ? 'bg-slate-950/80 border-slate-800 text-slate-400' : 'bg-slate-100 border-slate-200 text-slate-600'}`}>
              <th className="p-3">Record ID</th>
              <th className="p-3">Timestamp</th>
              <th className="p-3">Mental State</th>
              <th className="p-3">Confidence</th>
              <th className="p-3">GSR (µS)</th>
              <th className="p-3">PPG (BPM)</th>
              <th className="p-3">Facial Expression</th>
              <th className="p-3 text-right">Wellness</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/50 font-mono">
            {filteredHistory.length > 0 ? (
              filteredHistory.map((item) => {
                const meta = getMentalStateMeta(item.state);

                return (
                  <tr key={item.id} className={`hover:bg-cyan-500/5 transition-colors ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>
                    <td className="p-3 text-cyan-400">{item.id}</td>
                    <td className="p-3">{item.timestamp}</td>
                    <td className="p-3">
                      <span className={`px-2 py-0.5 rounded-full text-[10px] border font-bold ${meta.badgeBg} ${meta.badgeText}`}>
                        {item.state}
                      </span>
                    </td>
                    <td className="p-3">{item.confidence}%</td>
                    <td className="p-3">{item.gsr}</td>
                    <td className="p-3">{item.ppg}</td>
                    <td className="p-3">{item.emotion}</td>
                    <td className="p-3 text-right font-bold" style={{ color: meta.color }}>
                      {item.wellnessScore} / 100
                    </td>
                  </tr>
                );
              })
            ) : (
              <tr>
                <td colSpan={8} className="p-6 text-center text-slate-500 font-sans">
                  No log records match your filter criteria.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
