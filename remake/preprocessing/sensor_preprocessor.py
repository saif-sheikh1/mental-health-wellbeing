"""
MindSense AI — Sensor Data Preprocessor
========================================
Implements the complete 14-step preprocessing pipeline for multi-sensor
mental health data from Grove GSR, MAX30102 PPG, and NeuroSky TGAM EEG sensors.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from scipy.signal import butter, filtfilt, medfilt
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


class SensorPreprocessor:
    """Full preprocessing pipeline for multi-sensor mental health CSV data."""

    # Column groups
    EEG_COLS = [
        'eeg_delta', 'eeg_theta', 'eeg_low_alpha', 'eeg_high_alpha',
        'eeg_low_beta', 'eeg_high_beta', 'eeg_low_gamma', 'eeg_mid_gamma'
    ]

    ROBUST_COLS = [
        'gsr_raw', 'skin_conductance_level_uS', 'ppg_bpm', 'hrv_ms',
        'gsr_ppg_combined', 'autonomic_balance'
    ]

    DERIVED_EEG_POWER_COLS = [
        'alpha_power_total', 'beta_power_total', 'gamma_power_total'
    ]

    RATIO_COLS = [
        'theta_alpha_ratio', 'beta_alpha_ratio', 'engagement_index',
        'relaxation_index', 'stress_eeg_index'
    ]

    def __init__(self, config: dict):
        self.config = config
        self.sensor_csv = config['paths']['sensor_csv']
        self.models_dir = os.path.join(
            config['paths']['models_output_dir'], 'sensor_model'
        )
        os.makedirs(self.models_dir, exist_ok=True)

        self.gender_encoder = LabelEncoder()
        self.label_encoder = LabelEncoder()
        self.robust_scaler = RobustScaler()
        self.standard_scaler = StandardScaler()
        self.feature_names = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self):
        """Execute the full 14-step preprocessing pipeline.

        Returns
        -------
        dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
                        feature_names, label_encoder
        """
        logger.info("=" * 60)
        logger.info("SENSOR PREPROCESSING PIPELINE")
        logger.info("=" * 60)

        # STEP 1: Load CSV, drop participant_id
        logger.info("Step 1 — Loading CSV …")
        df = pd.read_csv(self.sensor_csv)
        logger.info(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        df = df.drop(columns=['participant_id'])

        # STEP 2: Encode gender
        logger.info("Step 2 — Encoding gender …")
        df['gender'] = self.gender_encoder.fit_transform(df['gender'])
        logger.info(f"  Gender classes: {list(self.gender_encoder.classes_)}")

        # STEP 3: Feature Engineering — 10 derived columns
        logger.info("Step 3 — Feature engineering (10 derived features) …")
        df = self._add_derived_features(df)

        # STEP 4: Butterworth low-pass filter on gsr_raw
        logger.info("Step 4 — Butterworth filter on GSR …")
        df['gsr_raw'] = self._butterworth_filter(df['gsr_raw'].values)

        # STEP 5: log1p on all 8 EEG columns
        logger.info("Step 5 — log1p transform on EEG columns …")
        for col in self.EEG_COLS:
            df[col] = np.log1p(df[col].clip(lower=0))

        # STEP 6: Clip PPG and HRV to physiological range
        logger.info("Step 6 — Clipping PPG [30,220] and HRV [5,200] …")
        df['ppg_bpm'] = self._median_filter_ppg(df['ppg_bpm'].values)
        df['ppg_bpm'] = df['ppg_bpm'].clip(30, 220)
        df['hrv_ms'] = df['hrv_ms'].clip(5, 200)

        # Separate target before scaling
        y = df['mental_state'].copy()
        df = df.drop(columns=['mental_state'])

        # STEP 6 (cont.): RobustScaler on physiological signals
        logger.info("Step 6 — RobustScaler on physiological columns …")
        df[self.ROBUST_COLS] = self.robust_scaler.fit_transform(
            df[self.ROBUST_COLS]
        )

        # STEP 7: StandardScaler on EEG + derived EEG power + ratio cols
        logger.info("Step 7 — StandardScaler on EEG & derived columns …")
        std_cols = self.EEG_COLS + self.DERIVED_EEG_POWER_COLS + self.RATIO_COLS
        df[std_cols] = self.standard_scaler.fit_transform(df[std_cols])

        # STEP 8: Encode mental_state labels
        logger.info("Step 8 — Encoding mental_state labels …")
        y_encoded = self.label_encoder.fit_transform(y)
        logger.info(f"  {len(self.label_encoder.classes_)} classes: "
                     f"{list(self.label_encoder.classes_)}")

        # Store feature names
        self.feature_names = list(df.columns)

        # STEP 9–12: Save all artifacts
        logger.info("Steps 9-12 — Saving preprocessing artifacts …")
        self._save_artifacts()

        # STEP 13: Train / Val / Test split (70/15/15)
        logger.info("Step 13 — Train/Val/Test split (70/15/15) …")
        X = df.values.astype(np.float32)

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_encoded, test_size=0.30, stratify=y_encoded, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
        )
        logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, "
                     f"Test: {len(X_test)}")

        # STEP 14: SMOTE on training set only
        logger.info("Step 14 — Applying SMOTE on training set …")
        smote = SMOTE(k_neighbors=5, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(f"  After SMOTE: {len(X_train)} training samples")

        logger.info("✅ Sensor preprocessing complete!")
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'label_encoder': self.label_encoder,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 10 derived feature columns."""
        eps = 1e-6

        df['theta_alpha_ratio'] = (
            df['eeg_theta']
            / (df['eeg_low_alpha'] + df['eeg_high_alpha'] + eps)
        )
        df['beta_alpha_ratio'] = (
            (df['eeg_low_beta'] + df['eeg_high_beta'])
            / (df['eeg_low_alpha'] + df['eeg_high_alpha'] + eps)
        )
        df['engagement_index'] = (
            df['eeg_low_beta']
            / (df['eeg_theta'] + df['eeg_low_alpha'] + eps)
        )
        df['relaxation_index'] = (
            df['eeg_high_alpha'] / (df['eeg_low_beta'] + eps)
        )
        df['stress_eeg_index'] = (
            (df['eeg_low_beta'] + df['eeg_high_beta'])
            / (df['eeg_theta'] + df['eeg_delta'] + eps)
        )
        df['gsr_ppg_combined'] = df['gsr_raw'] * df['ppg_bpm'] / 10000.0
        df['autonomic_balance'] = df['hrv_ms'] / (df['ppg_bpm'] + eps)

        df['alpha_power_total'] = df['eeg_low_alpha'] + df['eeg_high_alpha']
        df['beta_power_total'] = df['eeg_low_beta'] + df['eeg_high_beta']
        df['gamma_power_total'] = df['eeg_low_gamma'] + df['eeg_mid_gamma']

        return df

    def _butterworth_filter(self, signal: np.ndarray,
                            cutoff: float = 1.0,
                            fs: float = 10.0,
                            order: int = 4) -> np.ndarray:
        """Apply a Butterworth low-pass filter to simulate GSR denoising.

        Parameters
        ----------
        signal : array-like
            Raw GSR ADC values.
        cutoff : float
            Low-pass cutoff frequency in Hz (default 1 Hz).
        fs : float
            Assumed sampling frequency (simulated, since CSV rows aren't
            truly time-series — we treat them as sequential for filtering).
        order : int
            Filter order (default 4).
        """
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered = filtfilt(b, a, signal, padlen=min(3 * max(len(b), len(a)),
                                                      len(signal) - 1))
        return filtered

    def _median_filter_ppg(self, signal: np.ndarray,
                           kernel_size: int = 5) -> np.ndarray:
        """Apply a median filter to remove PPG spike artifacts."""
        return medfilt(signal, kernel_size=kernel_size)

    def _save_artifacts(self):
        """Save all preprocessing artifacts to disk."""
        # STEP 8: Label encoder
        joblib.dump(
            self.label_encoder,
            os.path.join(self.models_dir, 'label_encoder.pkl')
        )

        # STEP 9: RobustScaler
        joblib.dump(
            self.robust_scaler,
            os.path.join(self.models_dir, 'robust_scaler.pkl')
        )

        # STEP 10: StandardScaler
        joblib.dump(
            self.standard_scaler,
            os.path.join(self.models_dir, 'standard_scaler.pkl')
        )

        # STEP 11: Gender encoder
        joblib.dump(
            self.gender_encoder,
            os.path.join(self.models_dir, 'gender_encoder.pkl')
        )

        # STEP 12: Feature names
        with open(os.path.join(self.models_dir, 'feature_names.json'), 'w') as f:
            json.dump(self.feature_names, f, indent=2)

        logger.info(f"  Artifacts saved to {self.models_dir}")
