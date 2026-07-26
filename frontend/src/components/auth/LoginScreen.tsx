import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { BrainCircuit, Mail, Phone, User, ArrowRight, ShieldCheck, Activity, Sparkles } from 'lucide-react';
import { useDashboardStore } from '../../store/useDashboardStore';

export const LoginScreen: React.FC = () => {
  const { loginUser, isDarkMode } = useDashboardStore();
  const [name, setName] = useState('');
  const [phone, setPhone] = useState('');
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setError('Please enter your full name');
      return;
    }
    if (!phone.trim()) {
      setError('Please enter your phone number');
      return;
    }
    if (!email.trim() || !email.includes('@')) {
      setError('Please enter a valid email address');
      return;
    }

    setError('');
    setLoading(true);

    try {
      await loginUser(name, phone, email);
    } catch (err) {
      setError('Connection issue. Proceeding with offline patient session...');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`min-h-screen flex items-center justify-center p-4 relative overflow-hidden transition-colors duration-500 ${
      isDarkMode ? 'bg-slate-950 text-slate-100' : 'bg-slate-100 text-slate-900'
    }`}>
      {/* Background Ambient Glow Orbs */}
      <div className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full bg-cyan-500/10 blur-3xl animate-pulse" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 rounded-full bg-emerald-500/10 blur-3xl animate-pulse" />

      {/* Login Card */}
      <motion.div
        initial={{ opacity: 0, y: 20, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
        className={`w-full max-w-md p-8 rounded-3xl border relative z-10 shadow-2xl backdrop-blur-2xl ${
          isDarkMode 
            ? 'bg-slate-900/80 border-slate-800/80 shadow-cyan-950/20' 
            : 'bg-white/90 border-slate-200 shadow-slate-300/50'
        }`}
      >
        {/* Header Branding */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-tr from-cyan-600 via-teal-500 to-emerald-400 shadow-xl shadow-cyan-500/25 mb-4">
            <BrainCircuit className="w-8 h-8 text-white animate-pulse" />
          </div>
          <h1 className="text-2xl font-extrabold tracking-tight bg-gradient-to-r from-cyan-400 via-teal-300 to-emerald-400 bg-clip-text text-transparent">
            MindSense AI Portal
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Multimodal Mental Health & Biometric Diagnostic Intake
          </p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -6 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-3 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-xs font-medium text-center"
            >
              {error}
            </motion.div>
          )}

          {/* Name Field */}
          <div>
            <label className="block text-xs font-mono text-slate-400 mb-1.5">
              Full Patient Name <span className="text-cyan-400">*</span>
            </label>
            <div className="relative">
              <User className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
              <input
                type="text"
                placeholder="e.g. Alex Vance"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className={`w-full pl-10 pr-4 py-2.5 rounded-xl border text-xs outline-none transition-all ${
                  isDarkMode 
                    ? 'bg-slate-950/80 border-slate-800 text-slate-200 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/30' 
                    : 'bg-slate-50 border-slate-200 text-slate-900 focus:border-cyan-500'
                }`}
              />
            </div>
          </div>

          {/* Phone Field */}
          <div>
            <label className="block text-xs font-mono text-slate-400 mb-1.5">
              Phone Number <span className="text-cyan-400">*</span>
            </label>
            <div className="relative">
              <Phone className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
              <input
                type="tel"
                placeholder="+1 (555) 019-2834"
                value={phone}
                onChange={(e) => setPhone(e.target.value)}
                className={`w-full pl-10 pr-4 py-2.5 rounded-xl border text-xs outline-none transition-all ${
                  isDarkMode 
                    ? 'bg-slate-950/80 border-slate-800 text-slate-200 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/30' 
                    : 'bg-slate-50 border-slate-200 text-slate-900 focus:border-cyan-500'
                }`}
              />
            </div>
          </div>

          {/* Email Field */}
          <div>
            <label className="block text-xs font-mono text-slate-400 mb-1.5">
              Email Address <span className="text-cyan-400">*</span>
            </label>
            <div className="relative">
              <Mail className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
              <input
                type="email"
                placeholder="patient@mindsense.ai"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className={`w-full pl-10 pr-4 py-2.5 rounded-xl border text-xs outline-none transition-all ${
                  isDarkMode 
                    ? 'bg-slate-950/80 border-slate-800 text-slate-200 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/30' 
                    : 'bg-slate-50 border-slate-200 text-slate-900 focus:border-cyan-500'
                }`}
              />
            </div>
          </div>

          {/* Submit Button */}
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            type="submit"
            disabled={loading}
            className="w-full mt-2 py-3 px-4 rounded-xl bg-gradient-to-r from-cyan-500 via-teal-500 to-emerald-400 text-slate-950 font-bold text-sm shadow-lg shadow-cyan-500/25 flex items-center justify-center gap-2 hover:opacity-95 transition-opacity"
          >
            {loading ? (
              <span>Initializing Telemetry Session...</span>
            ) : (
              <>
                <span>Access Clinical Suite</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </motion.button>
        </form>

        {/* Security Footer */}
        <div className="mt-6 pt-4 border-t border-slate-800/60 flex items-center justify-between text-[11px] text-slate-500 font-mono">
          <span className="flex items-center gap-1 text-emerald-400">
            <ShieldCheck className="w-3.5 h-3.5" /> HIPAA Compliant Intake
          </span>
          <span>v2.4 Telemetry</span>
        </div>
      </motion.div>
    </div>
  );
};
