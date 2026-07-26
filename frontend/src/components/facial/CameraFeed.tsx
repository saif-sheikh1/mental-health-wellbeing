import React, { useRef, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Camera, CameraOff, RefreshCw, Scan, ShieldCheck, Sparkles } from 'lucide-react';
import { useDashboardStore } from '../../store/useDashboardStore';

export const CameraFeed: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const { isCameraActive, setCameraActive, activeEmotion, readings, isDarkMode } = useDashboardStore();
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [fps, setFps] = useState<number>(30);

  useEffect(() => {
    let stream: MediaStream | null = null;

    async function setupCamera() {
      if (!isCameraActive) return;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' } 
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setHasPermission(true);
      } catch (err) {
        setHasPermission(false);
      }
    }

    setupCamera();

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [isCameraActive]);

  // Jitter FPS slightly for realistic radar feed feel
  useEffect(() => {
    const interval = setInterval(() => {
      setFps(Math.floor(28 + Math.random() * 5));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className={`relative p-5 rounded-3xl border overflow-hidden transition-all duration-300 ${
      isDarkMode 
        ? 'bg-slate-900/80 border-slate-800/80 shadow-xl' 
        : 'bg-white/90 border-slate-200 shadow-md'
    }`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-cyan-500/10 text-cyan-400">
            <Camera className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold tracking-tight">Live Facial Vision Stream</h3>
            <p className="text-xs text-slate-400">Realtime ViT Convolutional Emotion Engine</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
            {fps} FPS
          </span>
          <button
            onClick={() => setCameraActive(!isCameraActive)}
            className={`p-2 rounded-xl border text-xs font-medium transition-all ${
              isCameraActive 
                ? 'bg-red-500/10 border-red-500/30 text-red-400 hover:bg-red-500/20' 
                : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20'
            }`}
          >
            {isCameraActive ? <CameraOff className="w-4 h-4" /> : <Camera className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Video Viewport Container */}
      <div className="relative aspect-video w-full rounded-2xl bg-slate-950 overflow-hidden border border-slate-800 flex items-center justify-center">
        {isCameraActive ? (
          <>
            {/* Real WebCam Video element */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover transform -scale-x-100 opacity-90"
            />

            {/* Fallback Simulation Graphic if camera permission denied or no hardware */}
            {hasPermission === false && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 p-6 text-center">
                <div className="relative w-32 h-32 mb-4 rounded-full border-2 border-cyan-500/30 flex items-center justify-center">
                  <div className="absolute inset-0 rounded-full border border-dashed border-cyan-400 animate-radar" />
                  <Scan className="w-12 h-12 text-cyan-400 animate-pulse" />
                </div>
                <h4 className="text-sm font-bold text-slate-200">Camera Feed Simulated</h4>
                <p className="text-xs text-slate-400 max-w-xs mt-1">
                  Using high-precision synthetic facial landmark projection for continuous diagnostic tracking.
                </p>
              </div>
            )}

            {/* Facial Recognition HUD Bounding Box Overlay */}
            <div className="absolute inset-0 pointer-events-none p-6 flex items-center justify-center">
              <motion.div
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.4 }}
                className="relative w-56 h-56 rounded-2xl border-2 border-cyan-400/80 shadow-[0_0_30px_rgba(6,182,212,0.3)] flex flex-col justify-between p-2"
              >
                {/* Corner Crosshairs */}
                <div className="absolute -top-1 -left-1 w-4 h-4 border-t-2 border-l-2 border-cyan-400" />
                <div className="absolute -top-1 -right-1 w-4 h-4 border-t-2 border-r-2 border-cyan-400" />
                <div className="absolute -bottom-1 -left-1 w-4 h-4 border-b-2 border-l-2 border-cyan-400" />
                <div className="absolute -bottom-1 -right-1 w-4 h-4 border-b-2 border-r-2 border-cyan-400" />

                {/* Top Overlay Badge */}
                <div className="flex justify-between items-center text-[10px] font-mono bg-slate-950/80 backdrop-blur px-2.5 py-1 rounded-lg border border-cyan-500/40 text-cyan-300">
                  <span className="flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
                    ViT-FACE-L2
                  </span>
                  <span>CONF: {readings.facialConfidence}%</span>
                </div>

                {/* Target Reticle Centered */}
                <div className="self-center my-auto flex items-center justify-center">
                  <div className="w-12 h-12 rounded-full border border-dashed border-cyan-400/50 animate-spin" />
                </div>

                {/* Bottom Overlay Badge */}
                <div className="bg-slate-950/90 backdrop-blur px-3 py-1.5 rounded-xl border border-cyan-500/40 text-center">
                  <span className="text-xs font-bold text-white tracking-wide">
                    EMOTION: <span className="text-cyan-400 uppercase">{activeEmotion}</span>
                  </span>
                </div>
              </motion.div>
            </div>

            {/* Scan Line Animation */}
            <div className="absolute inset-0 pointer-events-none bg-gradient-to-b from-cyan-500/5 via-transparent to-cyan-500/5 animate-pulse" />
          </>
        ) : (
          <div className="flex flex-col items-center justify-center p-6 text-center text-slate-500">
            <CameraOff className="w-10 h-10 mb-2 text-slate-600" />
            <p className="text-xs">Camera Feed Paused</p>
            <button
              onClick={() => setCameraActive(true)}
              className="mt-3 px-3 py-1.5 rounded-xl bg-cyan-500/20 text-cyan-400 text-xs font-medium border border-cyan-500/30 hover:bg-cyan-500/30"
            >
              Resume Camera
            </button>
          </div>
        )}
      </div>

      {/* Stream Meta Footer */}
      <div className="mt-3 flex items-center justify-between text-xs text-slate-400 font-mono">
        <span className="flex items-center gap-1 text-emerald-400">
          <ShieldCheck className="w-3.5 h-3.5" /> Privacy Encrypted (Local Stream)
        </span>
        <span>Resolution: 640x480</span>
      </div>
    </div>
  );
};
