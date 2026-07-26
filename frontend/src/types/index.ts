export type FacialEmotion = 
  | 'Happy' 
  | 'Sad' 
  | 'Angry' 
  | 'Neutral' 
  | 'Fear' 
  | 'Surprise' 
  | 'Disgust';

export type MentalState = 
  | 'NORMAL' 
  | 'LOW_STRESS' 
  | 'MODERATE_STRESS' 
  | 'LOW_ANXIETY'
  | 'HIGH_ANXIETY' 
  | 'PANIC_STATE' 
  | 'DEPRESSION' 
  | 'NO_SIGNAL';

export type ShortcutMode = 0 | 1 | 2 | 3 | 4 | 5 | null;

export interface EEGData {
  delta: number;
  theta: number;
  lowAlpha: number;
  highAlpha: number;
  lowBeta: number;
  highBeta: number;
  lowGamma: number;
  midGamma: number;
}

export interface SensorReadings {
  gsr: number;
  ppg: number;
  eeg: EEGData;
  facial: FacialEmotion;
  facialConfidence: number;
  facialProbabilities: Record<FacialEmotion, number>;
  lastUpdated: string;
}

export interface MentalStatePrediction {
  state: MentalState;
  confidence: number;
  wellnessScore: number;
  label: string;
  description: string;
  color: string;
  badgeBg: string;
  badgeText: string;
}

export interface RecommendationDetails {
  trend: 'improving' | 'worsening' | 'stable';
  trend_message: string;
  first_aid: string[];
  lifestyle: string[];
  professional: string[];
  emotion_tip: string;
}

export interface HistoryRecord {
  id: string;
  timestamp: string;
  state: MentalState;
  confidence: number;
  gsr: number;
  ppg: number;
  emotion: FacialEmotion;
  wellnessScore: number;
}

export interface PatientInfo {
  id: string;
  name: string;
  phone: string;
  email: string;
  age?: number;
  gender?: string;
  doctorName?: string;
  sessionTime?: string;
}

export type ActiveTab = 'master' | 'dashboard' | 'eeg' | 'sensors' | 'facial' | 'history';
