"""
MindSense AI — Future State Predictor Training
================================================
Seq2Seq model: Bidirectional LSTM Encoder + Bahdanau Attention + LSTM Decoder.

Purpose: Given last 10 consecutive sensor readings, predict the next 5 mental
states using teacher forcing with linear decay.

Since the CSV contains independent snapshots (not time-series), we generate
100,000 synthetic sequences using a 20×20 state transition matrix.
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.callbacks import EarlyStopping, TensorBoardLogger

logger = logging.getLogger(__name__)


# =========================================================================
# State Definitions & Transition Matrix
# =========================================================================

MENTAL_STATES = [
    'anxiety_high', 'anxiety_low', 'anxiety_moderate',
    'calm', 'cognitive_overload_high', 'cognitive_overload_low',
    'depression_high', 'depression_low', 'depression_moderate',
    'fatigue_high', 'fatigue_low', 'fatigue_moderate',
    'mixed_depression_fatigue', 'mixed_stress_anxiety',
    'normal', 'panic_state', 'relaxed',
    'stress_high', 'stress_low', 'stress_moderate',
]

STATE_SEVERITY = {
    'normal': 0, 'calm': 0, 'relaxed': 0,
    'fatigue_low': 1, 'depression_low': 1, 'stress_low': 1, 'anxiety_low': 1,
    'fatigue_moderate': 2, 'depression_moderate': 2, 'stress_moderate': 2,
    'anxiety_moderate': 2, 'cognitive_overload_low': 2,
    'fatigue_high': 3, 'depression_high': 3, 'stress_high': 3,
    'anxiety_high': 3, 'cognitive_overload_high': 3,
    'mixed_stress_anxiety': 3, 'mixed_depression_fatigue': 3,
    'panic_state': 4,
}


def build_transition_matrix():
    """Build a 20×20 state transition probability matrix.

    Rules:
      - Same severity: p=0.5 stay, p=0.3 worsen, p=0.2 improve
      - panic_state: p=0.6 stay, p=0.4 improve to anxiety_high
      - normal/calm/relaxed: p=0.7 stay, p=0.3 worsen to severity-1
    """
    n = len(MENTAL_STATES)
    T = np.zeros((n, n), dtype=np.float64)
    state_to_idx = {s: i for i, s in enumerate(MENTAL_STATES)}
    idx_to_state = {i: s for s, i in state_to_idx.items()}

    # Group states by severity
    severity_groups = {}
    for state, sev in STATE_SEVERITY.items():
        severity_groups.setdefault(sev, []).append(state)

    for i in range(n):
        state = idx_to_state[i]
        sev = STATE_SEVERITY[state]

        if state == 'panic_state':
            # Special: panic_state
            T[i, i] = 0.6
            T[i, state_to_idx['anxiety_high']] = 0.4
        elif sev == 0:
            # Normal/calm/relaxed: p=0.7 stay in sev-0, p=0.3 worsen
            same_group = severity_groups[0]
            for s in same_group:
                T[i, state_to_idx[s]] = 0.7 / len(same_group)
            # Worsen to severity 1
            if 1 in severity_groups:
                worse_group = severity_groups[1]
                for s in worse_group:
                    T[i, state_to_idx[s]] = 0.3 / len(worse_group)
        else:
            # General: p=0.5 stay in same severity, p=0.3 worsen, p=0.2 improve
            same_group = severity_groups.get(sev, [state])
            for s in same_group:
                T[i, state_to_idx[s]] += 0.5 / len(same_group)

            # Worsen
            if sev + 1 in severity_groups:
                worse_group = severity_groups[sev + 1]
                for s in worse_group:
                    T[i, state_to_idx[s]] += 0.3 / len(worse_group)
            else:
                # Can't worsen further, stay
                for s in same_group:
                    T[i, state_to_idx[s]] += 0.3 / len(same_group)

            # Improve
            if sev - 1 in severity_groups:
                better_group = severity_groups[sev - 1]
                for s in better_group:
                    T[i, state_to_idx[s]] += 0.2 / len(better_group)
            else:
                for s in same_group:
                    T[i, state_to_idx[s]] += 0.2 / len(same_group)

    # Row-normalize
    row_sums = T.sum(axis=1, keepdims=True)
    T = T / np.maximum(row_sums, 1e-10)

    return T


def generate_synthetic_sequences(data: dict, transition_matrix: np.ndarray,
                                 n_sequences: int = 100000,
                                 seq_len: int = 15,
                                 input_len: int = 10,
                                 noise_pct: float = 0.05,
                                 seed: int = 42):
    """Generate synthetic temporal sequences for Seq2Seq training.

    Parameters
    ----------
    data : dict
        Output of SensorPreprocessor.run() containing X_train, y_train, etc.
    transition_matrix : ndarray (20, 20)
    n_sequences : int
    seq_len : int — total sequence length (input + output)
    input_len : int — encoder input length
    noise_pct : float — temporal noise percentage on features

    Returns
    -------
    X_seq : ndarray (n_sequences, input_len, n_features)
    Y_seq : ndarray (n_sequences, seq_len - input_len)
    """
    rng = np.random.RandomState(seed)
    n_classes = transition_matrix.shape[0]
    output_len = seq_len - input_len

    X_train = data['X_train']
    y_train = data['y_train']
    n_features = X_train.shape[1]

    # Group samples by class for feature lookup
    class_indices = {}
    for cls in range(n_classes):
        idx = np.where(y_train == cls)[0]
        if len(idx) > 0:
            class_indices[cls] = idx
        else:
            class_indices[cls] = np.array([0])  # fallback

    X_seq = np.zeros((n_sequences, input_len, n_features), dtype=np.float32)
    Y_seq = np.zeros((n_sequences, output_len), dtype=np.int64)

    for i in range(n_sequences):
        # Start from a random state
        current_state = rng.randint(0, n_classes)
        states = [current_state]

        # Simulate seq_len steps using transition matrix
        for _ in range(seq_len - 1):
            probs = transition_matrix[current_state]
            next_state = rng.choice(n_classes, p=probs)
            states.append(next_state)
            current_state = next_state

        # X = first input_len steps as feature vectors + temporal noise
        for t in range(input_len):
            cls = states[t]
            sample_idx = rng.choice(class_indices[cls])
            features = X_train[sample_idx].copy()
            # Add ±5% temporal noise
            noise = rng.uniform(-noise_pct, noise_pct, size=features.shape)
            features = features * (1.0 + noise)
            X_seq[i, t] = features

        # Y = steps input_len to seq_len-1 class labels
        Y_seq[i] = states[input_len:seq_len]

    logger.info(f"  Generated {n_sequences} sequences "
                 f"(X: {X_seq.shape}, Y: {Y_seq.shape})")
    return X_seq, Y_seq


# =========================================================================
# Seq2Seq Model Components
# =========================================================================

class EncoderBiLSTM(nn.Module):
    """Bidirectional LSTM Encoder with LayerNorm."""

    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor (batch, seq_len, input_size)

        Returns
        -------
        outputs : Tensor (batch, seq_len, hidden_size*2)
        hidden  : tuple(h_n, c_n) — reshaped for decoder
        """
        outputs, (h_n, c_n) = self.lstm(x)
        outputs = self.layer_norm(outputs)

        # Combine bidirectional hidden states for decoder
        # h_n shape: (num_layers*2, batch, hidden)
        # We want: (num_layers, batch, hidden*2)
        batch_size = x.size(0)
        h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_size)
        h_n = torch.cat([h_n[:, 0], h_n[:, 1]], dim=2)  # (layers, batch, H*2)
        c_n = c_n.view(self.num_layers, 2, batch_size, self.hidden_size)
        c_n = torch.cat([c_n[:, 0], c_n[:, 1]], dim=2)

        return outputs, (h_n, c_n)


class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention mechanism."""

    def __init__(self, enc_hidden: int = 512, dec_hidden: int = 512):
        super().__init__()
        self.W1 = nn.Linear(enc_hidden, dec_hidden, bias=False)
        self.W2 = nn.Linear(dec_hidden, dec_hidden, bias=False)
        self.V = nn.Linear(dec_hidden, 1, bias=False)

    def forward(self, encoder_out, decoder_hidden):
        """
        Parameters
        ----------
        encoder_out : Tensor (batch, src_len, enc_hidden)
        decoder_hidden : Tensor (batch, dec_hidden)

        Returns
        -------
        context : Tensor (batch, enc_hidden)
        weights : Tensor (batch, src_len, 1)
        """
        # decoder_hidden: (batch, dec_hidden) → (batch, 1, dec_hidden)
        energy = torch.tanh(
            self.W1(encoder_out) + self.W2(decoder_hidden.unsqueeze(1))
        )
        attention_scores = self.V(energy)  # (batch, src_len, 1)
        weights = F.softmax(attention_scores, dim=1)
        context = (weights * encoder_out).sum(dim=1)  # (batch, enc_hidden)
        return context, weights


class DecoderLSTM(nn.Module):
    """LSTM Decoder with Bahdanau attention."""

    def __init__(self, output_size: int = 20, hidden_size: int = 512,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.attention = BahdanauAttention(hidden_size, hidden_size)
        self.lstm = nn.LSTM(
            output_size + hidden_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, input_token, hidden, encoder_out):
        """
        Parameters
        ----------
        input_token : Tensor (batch, output_size)  — one-hot or embedding
        hidden : tuple(h, c) for LSTM
        encoder_out : Tensor (batch, src_len, hidden_size)

        Returns
        -------
        prediction : Tensor (batch, output_size) — logits
        hidden : updated LSTM hidden state
        attn_weights : Tensor (batch, src_len, 1)
        """
        # Attention context from encoder outputs
        context, attn_weights = self.attention(
            encoder_out, hidden[0][-1]  # last layer's hidden
        )

        # Concatenate input + context → LSTM input
        rnn_input = torch.cat([input_token, context], dim=-1)
        rnn_input = rnn_input.unsqueeze(1)  # (batch, 1, out+hidden)

        output, hidden = self.lstm(rnn_input, hidden)
        prediction = self.fc_out(output.squeeze(1))

        return prediction, hidden, attn_weights


class Seq2SeqPredictor(nn.Module):
    """Full Seq2Seq model: BiLSTM Encoder → Attention → LSTM Decoder."""

    def __init__(self, input_size: int, output_size: int = 20,
                 hidden_size: int = 256, num_layers: int = 2,
                 dropout: float = 0.3, output_len: int = 5):
        super().__init__()
        self.output_size = output_size
        self.output_len = output_len

        self.encoder = EncoderBiLSTM(
            input_size, hidden_size, num_layers, dropout
        )
        self.decoder = DecoderLSTM(
            output_size, hidden_size * 2, num_layers, dropout
        )

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        """
        Parameters
        ----------
        src : Tensor (batch, input_len, input_size) — encoder input
        trg : Tensor (batch, output_len) — decoder target labels (optional)
        teacher_forcing_ratio : float — probability of using ground truth

        Returns
        -------
        outputs : Tensor (batch, output_len, output_size) — logits
        """
        batch_size = src.size(0)
        device = src.device

        encoder_out, hidden = self.encoder(src)

        # Decoder outputs container
        outputs = torch.zeros(
            batch_size, self.output_len, self.output_size, device=device
        )

        # First decoder input: zero vector
        dec_input = torch.zeros(batch_size, self.output_size, device=device)

        for t in range(self.output_len):
            prediction, hidden, _ = self.decoder(dec_input, hidden, encoder_out)
            outputs[:, t] = prediction

            # Teacher forcing
            if trg is not None and np.random.random() < teacher_forcing_ratio:
                # Use ground truth as next input (one-hot)
                dec_input = torch.zeros(
                    batch_size, self.output_size, device=device
                )
                dec_input.scatter_(1, trg[:, t].unsqueeze(1), 1.0)
            else:
                # Use predicted output (soft)
                dec_input = F.softmax(prediction, dim=1)

        return outputs


# =========================================================================
# Training
# =========================================================================

def train_future_predictor(config: dict, device: torch.device,
                           sensor_data: dict = None):
    """Train the Seq2Seq future state predictor.

    Parameters
    ----------
    config : dict — full training config
    device : torch.device
    sensor_data : dict — output from SensorPreprocessor.run().
                  If None, will re-run preprocessing.

    Returns
    -------
    model : Seq2SeqPredictor
    """
    logger.info("=" * 60)
    logger.info("FUTURE STATE PREDICTOR TRAINING")
    logger.info("=" * 60)

    fp_cfg = config['future_predictor']
    models_dir = os.path.join(config['paths']['models_output_dir'],
                               'future_predictor')
    os.makedirs(models_dir, exist_ok=True)

    # Build transition matrix
    logger.info("Building state transition matrix …")
    transition_matrix = build_transition_matrix()
    np.save(os.path.join(models_dir, 'transition_matrix.npy'),
            transition_matrix)
    logger.info(f"  Transition matrix saved ({transition_matrix.shape})")

    # Get sensor data if not provided
    if sensor_data is None:
        from preprocessing.sensor_preprocessor import SensorPreprocessor
        preprocessor = SensorPreprocessor(config)
        sensor_data = preprocessor.run()

    # Generate synthetic sequences
    logger.info("Generating synthetic sequences …")
    input_len = fp_cfg['seq_len_input']
    output_len = fp_cfg['seq_len_output']
    seq_len = input_len + output_len

    X_seq, Y_seq = generate_synthetic_sequences(
        sensor_data, transition_matrix,
        n_sequences=fp_cfg['n_sequences'],
        seq_len=seq_len,
        input_len=input_len,
    )

    # Train/Val split
    n_train = int(0.85 * len(X_seq))
    X_train_seq, X_val_seq = X_seq[:n_train], X_seq[n_train:]
    Y_train_seq, Y_val_seq = Y_seq[:n_train], Y_seq[n_train:]

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train_seq, dtype=torch.float32),
        torch.tensor(Y_train_seq, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_seq, dtype=torch.float32),
        torch.tensor(Y_val_seq, dtype=torch.long),
    )
    train_loader = DataLoader(
        train_ds, batch_size=fp_cfg['batch_size'],
        shuffle=True, num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds, batch_size=fp_cfg['batch_size'],
        shuffle=False, num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    # Model
    n_features = X_seq.shape[2]
    model = Seq2SeqPredictor(
        input_size=n_features,
        output_size=20,
        hidden_size=fp_cfg['hidden_size'],
        num_layers=fp_cfg['num_layers'],
        dropout=fp_cfg['dropout'],
        output_len=output_len,
    ).to(device)

    logger.info(f"  Seq2Seq params: "
                 f"{sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    optimizer = torch.optim.Adam(
        model.parameters(), lr=fp_cfg['learning_rate'],
        betas=(0.9, 0.98),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6,
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    early_stop = EarlyStopping(
        patience=fp_cfg['patience'], mode='min', restore_best=True
    )
    tb_logger = TensorBoardLogger(
        os.path.join(config['paths']['logs_dir'], 'future_predictor')
    )

    epochs = fp_cfg['epochs']
    tf_start = fp_cfg['teacher_forcing_start']
    tf_end = fp_cfg['teacher_forcing_end']

    for epoch in range(1, epochs + 1):
        # Linear decay of teacher forcing
        tf_ratio = tf_start - (tf_start - tf_end) * (epoch - 1) / max(epochs - 1, 1)

        # ── Train ──
        model.train()
        train_loss = 0.0
        train_total = 0

        pbar = tqdm(train_loader,
                     desc=f"  Seq2Seq Epoch {epoch}/{epochs}",
                     leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            with autocast(enabled=(device.type == 'cuda')):
                logits = model(xb, yb, teacher_forcing_ratio=tf_ratio)
                # logits: (batch, output_len, 20)
                # Flatten for CrossEntropyLoss
                loss = criterion(
                    logits.reshape(-1, 20),
                    yb.reshape(-1),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * xb.size(0)
            train_total += xb.size(0)
            pbar.set_postfix(loss=loss.item(), tf=f"{tf_ratio:.2f}")

        train_loss /= train_total

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_correct = 0
        val_steps = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                with autocast(enabled=(device.type == 'cuda')):
                    logits = model(xb, teacher_forcing_ratio=0.0)
                    loss = criterion(logits.reshape(-1, 20), yb.reshape(-1))
                val_loss += loss.item() * xb.size(0)
                val_total += xb.size(0)
                preds = logits.argmax(dim=2)  # (batch, output_len)
                val_correct += (preds == yb).sum().item()
                val_steps += yb.numel()

        val_loss /= val_total
        val_acc = val_correct / val_steps
        scheduler.step(val_loss)

        tb_logger.log_scalars('loss', {'train': train_loss, 'val': val_loss},
                               epoch)
        tb_logger.log_scalar('accuracy/val', val_acc, epoch)
        tb_logger.log_scalar('teacher_forcing', tf_ratio, epoch)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:3d} — train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"tf={tf_ratio:.2f}"
            )

        early_stop(epoch, val_loss, model)
        if early_stop.should_stop:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    early_stop.restore(model)
    tb_logger.close()

    # Save
    torch.save(model.state_dict(),
               os.path.join(models_dir, 'seq2seq_bilstm.pth'))
    pred_config = {
        'input_size': n_features,
        'output_size': 20,
        'hidden_size': fp_cfg['hidden_size'],
        'num_layers': fp_cfg['num_layers'],
        'dropout': fp_cfg['dropout'],
        'input_len': input_len,
        'output_len': output_len,
        'mental_states': MENTAL_STATES,
    }
    with open(os.path.join(models_dir, 'predictor_config.json'), 'w') as f:
        json.dump(pred_config, f, indent=2)

    logger.info(f"  ✅ Future predictor saved → {models_dir}")

    return model


if __name__ == '__main__':
    import yaml
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    with open('configs/training_config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_future_predictor(cfg, dev)
