import { create } from 'zustand';
import { 
  ActiveTab, 
  FacialEmotion, 
  HistoryRecord, 
  MentalState, 
  PatientInfo, 
  RecommendationDetails, 
  SensorReadings, 
  ShortcutMode 
} from '../types';
import { 
  calculateWellnessScore, 
  generateEmotionProbabilities, 
  generateSensorValuesForMode, 
  getRecommendationsForState,
  getRandomInRange, 
  SHORTCUT_RANGES 
} from '../utils/ranges';

interface DashboardState {
  isAuthenticated: boolean;
  patientInfo: PatientInfo;
  readings: SensorReadings;
  shortcutMode: ShortcutMode;
  activeEmotion: FacialEmotion;
  mentalState: MentalState;
  confidence: number;
  wellnessScore: number;
  recommendations: RecommendationDetails;
  history: HistoryRecord[];
  isDarkMode: boolean;
  activeTab: ActiveTab;
  isReportModalOpen: boolean;
  isBackendConnected: boolean;
  isCameraActive: boolean;
  isLivePolling: boolean;
  
  // Actions
  loginUser: (name: string, phone: string, email: string) => Promise<void>;
  logout: () => void;
  setShortcutMode: (mode: ShortcutMode) => void;
  setActiveEmotion: (emotion: FacialEmotion) => void;
  setDarkMode: (dark: boolean) => void;
  setActiveTab: (tab: ActiveTab) => void;
  setReportModalOpen: (open: boolean) => void;
  setCameraActive: (active: boolean) => void;
  setBackendConnected: (connected: boolean) => void;
  setLivePolling: (polling: boolean) => void;
  updateFromBackend: (data: any) => void;
  tickJitter: () => void;
}

const initialReadings = generateSensorValuesForMode(1, 'Happy');
const initialWellness = calculateWellnessScore('NORMAL', initialReadings);
const initialRecommendations = getRecommendationsForState('NORMAL', 'Happy');

const defaultPatient: PatientInfo = {
  id: 'PX-1001',
  name: 'Guest Patient',
  phone: '+1 (555) 019-2834',
  email: 'patient@mindsense.ai',
  doctorName: 'Dr. Elena Rostova, MD',
  sessionTime: 'Live Session'
};

const initialHistory: HistoryRecord[] = Array.from({ length: 8 }).map((_, i) => {
  const time = new Date(Date.now() - (7 - i) * 15000).toLocaleTimeString();
  return {
    id: `REC-${1000 + i}`,
    timestamp: time,
    state: i < 5 ? 'NORMAL' : 'MODERATE_STRESS',
    confidence: Math.round(75 + Math.random() * 20),
    gsr: Math.round(1400 + Math.random() * 800),
    ppg: Math.round(68 + Math.random() * 25),
    emotion: i % 2 === 0 ? 'Happy' : 'Neutral',
    wellnessScore: Math.round(70 + Math.random() * 25)
  };
});

export const useDashboardStore = create<DashboardState>((set, get) => ({
  isAuthenticated: false,
  patientInfo: defaultPatient,
  readings: initialReadings,
  shortcutMode: 1,
  activeEmotion: 'Happy',
  mentalState: 'NORMAL',
  confidence: 88,
  wellnessScore: initialWellness,
  recommendations: initialRecommendations,
  history: initialHistory,
  isDarkMode: true,
  activeTab: 'master',
  isReportModalOpen: false,
  isBackendConnected: true,
  isCameraActive: true,
  isLivePolling: true,

  loginUser: async (name: string, phone: string, email: string) => {
    try {
      const formData = new FormData();
      formData.append('name', name);
      formData.append('phone', phone);
      formData.append('email', email);
      
      const res = await fetch('/api/register', {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        const user = await res.json();
        set({
          isAuthenticated: true,
          patientInfo: {
            id: user.id || `PX-${Math.floor(1000 + Math.random() * 9000)}`,
            name: user.name || name,
            phone: user.phone || phone,
            email: user.email || email,
            doctorName: 'Dr. Elena Rostova, MD',
            sessionTime: new Date().toLocaleTimeString(),
          }
        });
        return;
      }
    } catch (e) {
      // Fallback local session
    }

    set({
      isAuthenticated: true,
      patientInfo: {
        id: `PX-${Math.floor(1000 + Math.random() * 9000)}`,
        name,
        phone,
        email,
        doctorName: 'Dr. Elena Rostova, MD',
        sessionTime: new Date().toLocaleTimeString(),
      }
    });
  },

  logout: () => set({ isAuthenticated: false }),

  setShortcutMode: (mode: ShortcutMode) => {
    if (mode === null) return;
    const def = SHORTCUT_RANGES[mode] ?? SHORTCUT_RANGES[1];
    const currentEmotion = get().activeEmotion;
    const newReadings = generateSensorValuesForMode(mode, currentEmotion);
    const targetState = def.state;
    const newWellness = calculateWellnessScore(targetState, newReadings);
    
    // Recommendations change STRICTLY according to state
    const recs = getRecommendationsForState(targetState, currentEmotion);

    const newRecord: HistoryRecord = {
      id: `REC-${Date.now().toString().slice(-4)}`,
      timestamp: new Date().toLocaleTimeString(),
      state: targetState,
      confidence: newReadings.facialConfidence || Math.round(getRandomInRange(75, 95, 0)),
      gsr: newReadings.gsr,
      ppg: newReadings.ppg,
      emotion: newReadings.facial,
      wellnessScore: newWellness,
    };

    set((state) => ({
      shortcutMode: mode,
      mentalState: targetState,
      readings: newReadings,
      confidence: mode === 0 ? 0 : (newReadings.facialConfidence || Math.round(getRandomInRange(78, 96, 0))),
      wellnessScore: newWellness,
      recommendations: recs,
      history: [newRecord, ...state.history.slice(0, 19)]
    }));
  },

  setActiveEmotion: (emotion: FacialEmotion) => {
    const { confidence, probabilities } = generateEmotionProbabilities(emotion);
    const isNoSignal = get().shortcutMode === 0;
    const currentState = get().mentalState;
    
    const recs = getRecommendationsForState(currentState, emotion);

    set((state) => ({
      activeEmotion: emotion,
      recommendations: recs,
      readings: {
        ...state.readings,
        facial: emotion,
        facialConfidence: isNoSignal ? 0 : confidence,
        facialProbabilities: isNoSignal ? state.readings.facialProbabilities : probabilities,
      }
    }));
  },

  setDarkMode: (dark: boolean) => set({ isDarkMode: dark }),
  setActiveTab: (tab: ActiveTab) => set({ activeTab: tab }),
  setReportModalOpen: (open: boolean) => set({ isReportModalOpen: open }),
  setCameraActive: (active: boolean) => set({ isCameraActive: active }),
  setBackendConnected: (connected: boolean) => set({ isBackendConnected: connected }),
  setLivePolling: (polling: boolean) => set({ isLivePolling: polling }),

  updateFromBackend: (data: any) => {
    if (!data) return;
    const currentShortcut = get().shortcutMode;

    let stateStr: MentalState;
    if (currentShortcut !== null) {
      stateStr = SHORTCUT_RANGES[currentShortcut]?.state || 'NORMAL';
    } else {
      stateStr = (data.state?.state as MentalState) || 'NORMAL';
    }

    const isNoSignal = currentShortcut === 0;

    if (isNoSignal) {
      set({
        mentalState: 'NO_SIGNAL',
        confidence: 0,
        wellnessScore: 0,
        readings: generateSensorValuesForMode(0),
        recommendations: getRecommendationsForState('NO_SIGNAL', get().activeEmotion),
      });
      return;
    }

    const def = SHORTCUT_RANGES[currentShortcut ?? 1] || SHORTCUT_RANGES[1];
    const conf = Math.round((data.state?.confidence || 0.88) * 100);
    const sensors = data.sensors || {};

    const gsrVal = (sensors.gsr && sensors.gsr > 0) 
      ? sensors.gsr 
      : Math.round(getRandomInRange(def.gsr[0], def.gsr[1], 0));

    const ppgVal = (sensors.ppg && sensors.ppg > 0) 
      ? sensors.ppg 
      : Math.round(getRandomInRange(def.ppg[0], def.ppg[1], 0));

    const updatedReadings: SensorReadings = {
      gsr: Math.max(1, gsrVal),
      ppg: Math.max(1, ppgVal),
      eeg: {
        delta: sensors.delta ?? get().readings.eeg.delta,
        theta: sensors.theta ?? get().readings.eeg.theta,
        lowAlpha: sensors.low_alpha ?? get().readings.eeg.lowAlpha,
        highAlpha: sensors.high_alpha ?? get().readings.eeg.highAlpha,
        lowBeta: sensors.low_beta ?? get().readings.eeg.lowBeta,
        highBeta: sensors.high_beta ?? get().readings.eeg.highBeta,
        lowGamma: sensors.low_gamma ?? get().readings.eeg.lowGamma,
        midGamma: sensors.mid_gamma ?? get().readings.eeg.midGamma,
      },
      facial: get().activeEmotion,
      facialConfidence: get().readings.facialConfidence,
      facialProbabilities: get().readings.facialProbabilities,
      lastUpdated: new Date().toLocaleTimeString(),
    };

    const wellness = calculateWellnessScore(stateStr, updatedReadings);
    
    // Only change recommendations if state changed or recommendations not set
    const currentRecs = get().recommendations;
    const recs = (get().mentalState !== stateStr || !currentRecs)
      ? getRecommendationsForState(stateStr, get().activeEmotion)
      : currentRecs;

    set((state) => ({
      mentalState: stateStr,
      confidence: conf,
      wellnessScore: wellness,
      readings: updatedReadings,
      isBackendConnected: true,
      recommendations: recs,
    }));
  },

  tickJitter: () => {
    const { shortcutMode, readings, activeEmotion, mentalState, recommendations } = get();

    if (shortcutMode === 0) {
      const zeroReadings = generateSensorValuesForMode(0);
      set({
        mentalState: 'NO_SIGNAL',
        confidence: 0,
        wellnessScore: 0,
        readings: zeroReadings,
        recommendations: getRecommendationsForState('NO_SIGNAL', activeEmotion),
      });
      return;
    }

    const currentMode = shortcutMode ?? 1;
    const def = SHORTCUT_RANGES[currentMode] || SHORTCUT_RANGES[1];
    const stickyState = def.state;
    
    // Fluctuate telemetry values ONLY, keep recommendations stable for this state!
    const newGSR = Math.max(1, Math.round(getRandomInRange(def.gsr[0], def.gsr[1], 0)));
    const newPPG = Math.max(1, Math.round(getRandomInRange(def.ppg[0], def.ppg[1], 0)));
    const newEEG = {
      delta: getRandomInRange(def.delta[0], def.delta[1], 3),
      theta: getRandomInRange(def.theta[0], def.theta[1], 3),
      lowAlpha: getRandomInRange(def.lowAlpha[0], def.lowAlpha[1], 3),
      highAlpha: getRandomInRange(def.highAlpha[0], def.highAlpha[1], 3),
      lowBeta: getRandomInRange(def.lowBeta[0], def.lowBeta[1], 3),
      highBeta: getRandomInRange(def.highBeta[0], def.highBeta[1], 3),
      lowGamma: getRandomInRange(def.lowGamma[0], def.lowGamma[1], 3),
      midGamma: getRandomInRange(def.midGamma[0], def.midGamma[1], 3),
    };

    const { confidence, probabilities } = generateEmotionProbabilities(activeEmotion);
    const updated: SensorReadings = {
      ...readings,
      gsr: newGSR,
      ppg: newPPG,
      eeg: newEEG,
      facialConfidence: confidence,
      facialProbabilities: probabilities,
      lastUpdated: new Date().toLocaleTimeString(),
    };

    const wellness = calculateWellnessScore(stickyState, updated);

    // Keep recommendations STABLE and UNCHANGED during telemetry jitter ticks unless state changed
    const recs = (mentalState !== stickyState)
      ? getRecommendationsForState(stickyState, activeEmotion)
      : recommendations;

    set({
      mentalState: stickyState,
      confidence: confidence,
      readings: updated,
      wellnessScore: wellness,
      recommendations: recs,
    });
  }
}));
