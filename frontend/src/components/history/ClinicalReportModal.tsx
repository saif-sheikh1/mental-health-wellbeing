import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Printer, FileText, CheckCircle2, ShieldCheck, Download } from 'lucide-react';
import { useDashboardStore } from '../../store/useDashboardStore';
import { getMentalStateMeta } from '../../utils/ranges';

export const ClinicalReportModal: React.FC = () => {
  const { isReportModalOpen, setReportModalOpen, patientInfo, mentalState, confidence, wellnessScore, readings } = useDashboardStore();
  const meta = getMentalStateMeta(mentalState);

  if (!isReportModalOpen) return null;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="relative w-full max-w-3xl bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-2xl text-slate-100 max-h-[90vh] overflow-y-auto"
        >
          {/* Close Button */}
          <button
            onClick={() => setReportModalOpen(false)}
            className="absolute top-4 right-4 p-2 rounded-xl bg-slate-800 text-slate-400 hover:text-white"
          >
            <X className="w-5 h-5" />
          </button>

          {/* Report Document Header */}
          <div className="border-b border-slate-800 pb-4 mb-4 flex items-center justify-between">
            <div>
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-cyan-400" />
                <h2 className="text-xl font-bold">MindSense AI — Patient Telemetry Report</h2>
              </div>
              <p className="text-xs text-slate-400 font-mono mt-0.5">
                CONFIDENTIAL MEDICAL ASSESSMENT • GENERATED {new Date().toLocaleDateString()}
              </p>
            </div>
            <div className="text-right font-mono text-xs text-emerald-400">
              <span className="flex items-center gap-1">
                <ShieldCheck className="w-4 h-4" /> Authenticated Signature
              </span>
            </div>
          </div>

          {/* Patient Details */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-3.5 rounded-2xl bg-slate-950/70 border border-slate-800 text-xs font-mono mb-4">
            <div>
              <span className="text-slate-500 block">Patient Name</span>
              <span className="font-bold text-white">{patientInfo.name}</span>
            </div>
            <div>
              <span className="text-slate-500 block">Patient ID</span>
              <span className="text-cyan-400">{patientInfo.id}</span>
            </div>
            <div>
              <span className="text-slate-500 block">Attending Physician</span>
              <span className="text-slate-300">{patientInfo.doctorName}</span>
            </div>
            <div>
              <span className="text-slate-500 block">Session</span>
              <span className="text-slate-300">{new Date().toLocaleTimeString()}</span>
            </div>
          </div>

          {/* Diagnostic Summary Box */}
          <div className="p-4 rounded-2xl border bg-slate-950/80 border-slate-800 space-y-2 mb-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-slate-400">Primary Mental State Diagnosis</span>
              <span className={`px-2.5 py-0.5 rounded-full text-xs font-mono font-bold border ${meta.badgeBg} ${meta.badgeText}`}>
                {mentalState}
              </span>
            </div>
            <p className="text-sm font-bold text-slate-200">{meta.label}</p>
            <p className="text-xs text-slate-400 leading-relaxed">{meta.description}</p>
            
            <div className="flex items-center justify-between text-xs font-mono pt-2 border-t border-slate-800">
              <span>AI Classifier Confidence: <strong className="text-cyan-400">{confidence}%</strong></span>
              <span>Calculated Wellness Index: <strong style={{ color: meta.color }}>{wellnessScore} / 100</strong></span>
            </div>
          </div>

          {/* Biometrics Table */}
          <div className="space-y-2 mb-6">
            <h4 className="text-xs font-mono uppercase text-slate-400 tracking-wider">Physiological Sensor Telemetry</h4>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-slate-500 block">GSR Electrodermal</span>
                <span className="font-bold text-amber-400 text-sm">{readings.gsr} µS</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-slate-500 block">PPG Heart Rate</span>
                <span className="font-bold text-rose-400 text-sm">{readings.ppg} BPM</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-slate-500 block">Facial Expression</span>
                <span className="font-bold text-teal-400 text-sm">{readings.facial} ({readings.facialConfidence}%)</span>
              </div>
              <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
                <span className="text-slate-500 block">High Alpha EEG</span>
                <span className="font-bold text-cyan-400 text-sm">{readings.eeg.highAlpha.toFixed(2)} µV</span>
              </div>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center justify-end gap-3 pt-4 border-t border-slate-800">
            <button
              onClick={() => window.print()}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold"
            >
              <Printer className="w-4 h-4" /> Print PDF
            </button>
            <button
              onClick={() => setReportModalOpen(false)}
              className="px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs"
            >
              Done & Close
            </button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
};
