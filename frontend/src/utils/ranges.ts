import { FacialEmotion, MentalState, RecommendationDetails, SensorReadings } from '../types';

export interface ShortcutRangeDefinition {
  state: MentalState;
  label: string;
  gsr: [number, number];
  ppg: [number, number];
  delta: [number, number];
  theta: [number, number];
  lowAlpha: [number, number];
  highAlpha: [number, number];
  lowBeta: [number, number];
  highBeta: [number, number];
  lowGamma: [number, number];
  midGamma: [number, number];
  defaultEmotion: FacialEmotion;
}

// Strict PPG and GSR ranges as requested
export const SHORTCUT_RANGES: Record<number, ShortcutRangeDefinition> = {
  0: {
    state: 'NO_SIGNAL',
    label: 'No Signal',
    gsr: [0, 0],
    ppg: [0, 0],
    delta: [0, 0],
    theta: [0, 0],
    lowAlpha: [0, 0],
    highAlpha: [0, 0],
    lowBeta: [0, 0],
    highBeta: [0, 0],
    lowGamma: [0, 0],
    midGamma: [0, 0],
    defaultEmotion: 'Neutral',
  },
  1: {
    state: 'NORMAL',
    label: 'Normal',
    gsr: [1200, 1800],
    ppg: [60, 80],
    delta: [0.20, 0.35],
    theta: [0.25, 0.45],
    lowAlpha: [0.65, 0.85],
    highAlpha: [0.70, 0.90],
    lowBeta: [0.20, 0.40],
    highBeta: [0.20, 0.35],
    lowGamma: [0.15, 0.30],
    midGamma: [0.15, 0.30],
    defaultEmotion: 'Happy',
  },
  2: {
    state: 'MODERATE_STRESS',
    label: 'Moderate Stress',
    gsr: [2200, 2600],
    ppg: [90, 105],
    delta: [0.20, 0.30],
    theta: [0.25, 0.40],
    lowAlpha: [0.25, 0.40],
    highAlpha: [0.25, 0.40],
    lowBeta: [0.55, 0.70],
    highBeta: [0.60, 0.75],
    lowGamma: [0.35, 0.50],
    midGamma: [0.45, 0.60],
    defaultEmotion: 'Neutral',
  },
  3: {
    state: 'LOW_ANXIETY',
    label: 'Low Anxiety',
    gsr: [2600, 3000],
    ppg: [105, 120],
    delta: [0.15, 0.25],
    theta: [0.20, 0.35],
    lowAlpha: [0.20, 0.35],
    highAlpha: [0.20, 0.35],
    lowBeta: [0.60, 0.75],
    highBeta: [0.75, 0.90],
    lowGamma: [0.55, 0.70],
    midGamma: [0.60, 0.75],
    defaultEmotion: 'Fear',
  },
  4: {
    state: 'PANIC_STATE',
    label: 'Panic State',
    gsr: [3000, 4095],
    ppg: [120, 150],
    delta: [0.20, 0.35],
    theta: [0.30, 0.45],
    lowAlpha: [0.10, 0.25],
    highAlpha: [0.10, 0.25],
    lowBeta: [0.70, 0.90],
    highBeta: [0.85, 1.00],
    lowGamma: [0.70, 0.90],
    midGamma: [0.85, 1.00],
    defaultEmotion: 'Surprise',
  },
  5: {
    state: 'DEPRESSION',
    label: 'Low Depression',
    gsr: [1000, 1400],
    ppg: [55, 65],
    delta: [0.45, 0.70],
    theta: [0.55, 0.80],
    lowAlpha: [0.15, 0.30],
    highAlpha: [0.15, 0.30],
    lowBeta: [0.20, 0.35],
    highBeta: [0.20, 0.35],
    lowGamma: [0.15, 0.30],
    midGamma: [0.15, 0.30],
    defaultEmotion: 'Sad',
  },
};

export const ALL_EMOTIONS: FacialEmotion[] = [
  'Happy', 'Sad', 'Angry', 'Neutral', 'Fear', 'Surprise', 'Disgust'
];

export function getRandomInRange(min: number, max: number, decimals: number = 2): number {
  if (min === max) return min;
  const val = Math.random() * (max - min) + min;
  const factor = Math.pow(10, decimals);
  return Math.round(val * factor) / factor;
}

export function generateEmotionProbabilities(detected: FacialEmotion): {
  confidence: number;
  probabilities: Record<FacialEmotion, number>;
} {
  const primaryConf = getRandomInRange(62, 88, 1);
  let remaining = 100 - primaryConf;

  const otherEmotions = ALL_EMOTIONS.filter(e => e !== detected);
  const rawShares = otherEmotions.map(() => Math.random() * Math.random());
  const totalShares = rawShares.reduce((a, b) => a + b, 0);

  const probs: Record<string, number> = {};
  probs[detected] = primaryConf;

  let assignedSum = primaryConf;
  otherEmotions.forEach((e, idx) => {
    if (idx === otherEmotions.length - 1) {
      const share = Math.max(0.5, Math.round((100 - assignedSum) * 10) / 10);
      probs[e] = share;
    } else {
      const share = Math.max(0.5, Math.round((remaining * (rawShares[idx] / totalShares)) * 10) / 10);
      probs[e] = share;
      assignedSum += share;
    }
  });

  return {
    confidence: primaryConf,
    probabilities: probs as Record<FacialEmotion, number>,
  };
}

export function generateSensorValuesForMode(mode: number, currentEmotion?: FacialEmotion): SensorReadings {
  const def = SHORTCUT_RANGES[mode] ?? SHORTCUT_RANGES[1];
  const emotion = currentEmotion || def.defaultEmotion;
  const { confidence, probabilities } = generateEmotionProbabilities(emotion);

  const isNoSignal = mode === 0;

  return {
    gsr: isNoSignal ? 0 : Math.round(getRandomInRange(def.gsr[0], def.gsr[1], 0)),
    ppg: isNoSignal ? 0 : Math.round(getRandomInRange(def.ppg[0], def.ppg[1], 0)),
    eeg: {
      delta: isNoSignal ? 0 : getRandomInRange(def.delta[0], def.delta[1], 3),
      theta: isNoSignal ? 0 : getRandomInRange(def.theta[0], def.theta[1], 3),
      lowAlpha: isNoSignal ? 0 : getRandomInRange(def.lowAlpha[0], def.lowAlpha[1], 3),
      highAlpha: isNoSignal ? 0 : getRandomInRange(def.highAlpha[0], def.highAlpha[1], 3),
      lowBeta: isNoSignal ? 0 : getRandomInRange(def.lowBeta[0], def.lowBeta[1], 3),
      highBeta: isNoSignal ? 0 : getRandomInRange(def.highBeta[0], def.highBeta[1], 3),
      lowGamma: isNoSignal ? 0 : getRandomInRange(def.lowGamma[0], def.lowGamma[1], 3),
      midGamma: isNoSignal ? 0 : getRandomInRange(def.midGamma[0], def.midGamma[1], 3),
    },
    facial: isNoSignal ? 'Neutral' : emotion,
    facialConfidence: isNoSignal ? 0 : confidence,
    facialProbabilities: isNoSignal 
      ? ALL_EMOTIONS.reduce((acc, e) => ({ ...acc, [e]: e === 'Neutral' ? 100 : 0 }), {} as Record<FacialEmotion, number>)
      : probabilities,
    lastUpdated: new Date().toLocaleTimeString(),
  };
}

// Deterministic, state-locked recommendations that only change when the mental state changes
export function getRecommendationsForState(state: MentalState, emotion: FacialEmotion = 'Neutral'): RecommendationDetails {
  const normState = (state || 'NORMAL').toUpperCase() as MentalState;
  const normEmo = (emotion || 'Neutral').toLowerCase();

  const emotionTips: Record<string, string> = {
    happy: 'Positive affect detected. High cognitive clarity & optimal alpha coherence.',
    sad: 'Subdued affect detected. Gentle kinetic movement & light therapy recommended.',
    angry: 'High arousal affect detected. Slow abdominal exhalations advised.',
    neutral: 'Calm baseline affect detected. Ideal state for sustained focus & learning.',
    fear: 'Visible anxiety cues detected. Extended exhalations will activate parasympathetic tone.',
    surprise: 'Heightened startle response detected. Pause and anchor sensory focus.',
    disgust: 'Negative affect detected. A brief physical environment reset will help.',
  };

  switch (normState) {
    case 'PANIC_STATE':
      return {
        trend: 'worsening',
        trend_message: 'Critical sympathetic surge & tachycardia spike detected. Vagal nerve stimulation required.',
        first_aid: [
          'CRITICAL VAGAL RESET: Apply cold compress or ice pack directly to cervical spine / nape of neck',
          'Extended Exhalation Protocol: Inhale through nose for 3s, slow mouth exhale for 8s (repeat 5x)',
          'Tactile Anchoring: Place both feet flat on floor and press palms together firmly'
        ],
        lifestyle: [
          'Immediate Sensory Reduction: Transition to quiet, dimly-lit space',
          'Sip 200ml cold water slowly in small measured swallowing cycles',
          'Restrict screen stimulation & loud auditory inputs for 2 hours'
        ],
        professional: [
          'IMPORTANT: If chest pressure, numbness, or dyspnea persists, seek emergency medical evaluation',
          'Export telemetry log for urgent physician evaluation'
        ],
        emotion_tip: emotionTips[normEmo] || 'High sympatheto-vagal imbalance detected. Priority vagal grounding required.'
      };

    case 'LOW_ANXIETY':
    case 'HIGH_ANXIETY':
      return {
        trend: 'worsening',
        trend_message: 'High beta/gamma power dominance with elevated heart rate. Heightened vigilance trend.',
        first_aid: [
          '5-4-3-2-1 Sensory Grounding: Identify 5 visible items, 4 physical textures, 3 ambient sounds',
          'Diaphragmatic Breathing: 4s deep belly inhale, 7s hold, 8s slow pursed-lip exhale',
          'Progressive Muscle Release: Squeeze fist tight for 5s, then release completely'
        ],
        lifestyle: [
          'Strict Stimulant Fast: Avoid caffeine, nicotine, and energy supplements for rest of day',
          'Acoustic Entrainment: Listen to 432Hz alpha-wave audio for 15-20 minutes',
          'Digital Wind-Down: Avoid high-stimulation news or social media streams'
        ],
        professional: [
          'Consider Cognitive Behavioral Therapy (CBT) grounding strategies if anxiety episodes recur',
          'Correlate weekly heart rate variability (HRV) logs with stress events'
        ],
        emotion_tip: emotionTips[normEmo] || 'Elevated arousal detected. 4-7-8 breathing recommended to lower heart rate.'
      };

    case 'MODERATE_STRESS':
      return {
        trend: 'worsening',
        trend_message: 'Elevated skin conductance (GSR) and heart rate reduction in HRV noted.',
        first_aid: [
          'Box Breathing 4-4-4-4: 4s Inhale, 4s Hold, 4s Exhale, 4s Hold (Repeat 4 Cycles)',
          'Posture Reset: Stand up, stretch arms overhead, and roll shoulders back 5 times',
          'Hydration Break: Drink 250ml electrolyte-rich water'
        ],
        lifestyle: [
          'Cognitive Offloading: Journal current top 3 stressors for 5 minutes before bed',
          'Aerobic Micro-Burst: Schedule 20-30 minutes of moderate morning cardio tomorrow',
          'Magnesium Supplementation: Consider dietary magnesium to support muscle relaxation'
        ],
        professional: [
          'Monitor electrodermal stress trend across next 7 days of monitoring',
          'Schedule routine check-in if moderate stress persists >3 consecutive days'
        ],
        emotion_tip: emotionTips[normEmo] || 'Moderate stress detected. A brief 5-minute outdoor walk will reset cortisol.'
      };

    case 'LOW_STRESS':
      return {
        trend: 'stable',
        trend_message: 'Mild electrodermal arousal detected. Early cognitive fatigue or mild workload.',
        first_aid: [
          'Ergonomic Break: Step away from screen for 2 minutes and gaze at distance >20 feet',
          'Deep Diaphragmatic Breath: Take 3 deep abdominal inhalations',
          'Masseter Release: Gently massage jaw muscles to release facial tension'
        ],
        lifestyle: [
          'Circadian Sun Exposure: Get 10-15 minutes of direct morning sunlight',
          'Limit evening caffeine intake past 2:00 PM',
          'Engage in 15 minutes of light evening yoga or stretching'
        ],
        professional: [
          'Biometric trend remains well within manageable clinical thresholds',
          'Continue routine wearable tracking'
        ],
        emotion_tip: emotionTips[normEmo] || 'Mild stress detected. Brief shoulder roll and eye break recommended.'
      };

    case 'DEPRESSION':
      return {
        trend: 'worsening',
        trend_message: 'Hypoarousal with dominant slow-wave delta/theta activity & low physiological reactivity.',
        first_aid: [
          'Bright Phototherapy: Expose eyes to bright natural light or 10,000-lux therapy lamp (15 mins)',
          'Kinesthetic Activation: Perform 5 minutes of light rhythmic body movements or jumping jacks',
          'Auditory Stimulation: Play upbeat high-BPM music to stimulate cortical arousal'
        ],
        lifestyle: [
          'Consistent Wake Schedule: Maintain identical morning wake time daily',
          'Micro-Goal Execution: Complete one small, achievable physical task (e.g. bed making, short walk)',
          'Nutritional Support: Prioritize omega-3 fatty acids and protein-dense meals'
        ],
        professional: [
          'Consult a healthcare professional or licensed therapist if low mood & hypoarousal persist',
          'Track mood & EEG delta power trend weekly'
        ],
        emotion_tip: emotionTips[normEmo] || 'Hypoarousal detected. Light therapy & rhythmic physical movement recommended.'
      };

    case 'NO_SIGNAL':
      return {
        trend: 'stable',
        trend_message: 'Telemetry leads offline or awaiting sensor transmission.',
        first_aid: [
          'Verify electrode lead contact with skin surfaces',
          'Check receiver connection on USB / Bluetooth port'
        ],
        lifestyle: [
          'Ensure biometric dry sensors are clean and dry before session'
        ],
        professional: [
          'Technical telemetry diagnostics available'
        ],
        emotion_tip: 'Sensors offline. Connect hardware or press keyboard keys to preview.'
      };

    case 'NORMAL':
    default:
      return {
        trend: 'improving',
        trend_message: 'Optimal homeostatic balance. High alpha brainwave coherence & healthy heart rate.',
        first_aid: [
          'Maintain relaxed posture and rhythmic abdominal breathing',
          'Hydrate with 250ml pure water to support optimal cellular function',
          'Anchor baseline state with 1 minute of gratitude reflection'
        ],
        lifestyle: [
          'Maintain 7.5 - 8.0 hours of high-quality sleep hygiene',
          'Schedule 20-30 minutes of daily outdoor aerobic activity',
          'Practice 10 minutes of daily mindfulness meditation'
        ],
        professional: [
          'Biometric telemetry indicates optimal health baseline',
          'Routine clinical screening recommended every 30 days'
        ],
        emotion_tip: emotionTips[normEmo] || 'Optimal baseline detected. Excellent state for high-focus cognitive tasks.'
      };
  }
}

export function calculateWellnessScore(state: MentalState, readings: SensorReadings): number {
  if (state === 'NO_SIGNAL') return 0;
  if (state === 'NORMAL') return Math.min(98, Math.max(82, Math.round(88 + (readings.eeg.highAlpha * 10) - (readings.ppg / 20))));
  if (state === 'LOW_STRESS') return Math.min(80, Math.max(65, Math.round(72 - (readings.gsr / 400))));
  if (state === 'MODERATE_STRESS') return Math.min(64, Math.max(48, Math.round(56 - (readings.ppg / 10))));
  if (state === 'LOW_ANXIETY' || state === 'HIGH_ANXIETY') return Math.min(47, Math.max(32, Math.round(40 - (readings.gsr / 500))));
  if (state === 'PANIC_STATE') return Math.min(30, Math.max(12, Math.round(22 - (readings.ppg / 15))));
  if (state === 'DEPRESSION') return Math.min(45, Math.max(25, Math.round(35 + (readings.eeg.delta * 15))));
  return 75;
}

export function getMentalStateMeta(state: MentalState) {
  switch (state) {
    case 'NORMAL':
      return {
        label: 'Normal',
        description: 'Balanced sympathetic & parasympathetic tone. High alpha brainwave coherence.',
        color: '#10b981',
        badgeBg: 'bg-emerald-500/15 border-emerald-500/30',
        badgeText: 'text-emerald-400',
      };
    case 'LOW_STRESS':
      return {
        label: 'Low Stress',
        description: 'Slight elevation in skin conductance. Mild cognitive fatigue detected.',
        color: '#84cc16',
        badgeBg: 'bg-lime-500/15 border-lime-500/30',
        badgeText: 'text-lime-400',
      };
    case 'MODERATE_STRESS':
      return {
        label: 'Moderate Stress',
        description: 'Elevated heart rate & electrodermal arousal. Alpha power suppression noted.',
        color: '#f59e0b',
        badgeBg: 'bg-amber-500/15 border-amber-500/30',
        badgeText: 'text-amber-400',
      };
    case 'LOW_ANXIETY':
    case 'HIGH_ANXIETY':
      return {
        label: 'Low Anxiety',
        description: 'High beta/gamma dominance. Tachycardia trend with heightened vigilance.',
        color: '#f97316',
        badgeBg: 'bg-orange-500/15 border-orange-500/30',
        badgeText: 'text-orange-400',
      };
    case 'PANIC_STATE':
      return {
        label: 'Panic State',
        description: 'Extreme sympathetic surge. Critical GSR spikes and tachycardia present.',
        color: '#ef4444',
        badgeBg: 'bg-red-500/15 border-red-500/30',
        badgeText: 'text-red-400 animate-pulse',
      };
    case 'DEPRESSION':
      return {
        label: 'Low Depression',
        description: 'Dominant slow delta/theta activity with low physiological responsiveness.',
        color: '#8b5cf6',
        badgeBg: 'bg-purple-500/15 border-purple-500/30',
        badgeText: 'text-purple-400',
      };
    case 'NO_SIGNAL':
    default:
      return {
        label: 'No Signal',
        description: 'Sensory leads offline or awaiting valid biometric transmission.',
        color: '#64748b',
        badgeBg: 'bg-slate-500/15 border-slate-500/30',
        badgeText: 'text-slate-400',
      };
  }
}
