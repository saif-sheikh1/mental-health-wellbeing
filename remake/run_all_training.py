"""
MindSense AI — Master Training Runner
=======================================
Orchestrates the full multi-modal mental health detection training pipeline.

Usage:
    python run_all_training.py --config configs/training_config.yaml
    python run_all_training.py --config configs/training_config.yaml --skip-facial
    python run_all_training.py --config configs/training_config.yaml --eval-only
    python run_all_training.py --config configs/training_config.yaml --gpu
"""

import argparse
import logging
import os
import sys
import time
import random
import numpy as np
import torch
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(args, config):
    """Determine compute device from args and config."""
    if args.cpu:
        return torch.device('cpu')
    if args.gpu:
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            logging.warning("--gpu specified but CUDA not available. "
                            "Falling back to CPU.")
            return torch.device('cpu')

    device_cfg = config['training'].get('device', 'auto')
    if device_cfg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_cfg)


def format_time(seconds: float) -> str:
    """Format elapsed time as HH:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(
        description="MindSense AI — Master Training Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_training.py --config configs/training_config.yaml
  python run_all_training.py --config configs/training_config.yaml --gpu
  python run_all_training.py --config configs/training_config.yaml --skip-facial
  python run_all_training.py --config configs/training_config.yaml --eval-only
        """,
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to training_config.yaml'
    )
    parser.add_argument(
        '--gpu', action='store_true',
        help='Force GPU usage (error if CUDA unavailable)'
    )
    parser.add_argument(
        '--cpu', action='store_true',
        help='Force CPU usage'
    )
    parser.add_argument(
        '--skip-facial', action='store_true',
        help='Skip facial model training'
    )
    parser.add_argument(
        '--skip-fusion', action='store_true',
        help='Skip fusion model training'
    )
    parser.add_argument(
        '--eval-only', action='store_true',
        help='Only run evaluation (models already trained)'
    )
    args = parser.parse_args()

    # ── Load Config ──
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Make paths relative to project root
    # Ensure UTF-8 output on Windows console
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    # Create output directories
    os.makedirs(config['paths']['models_output_dir'], exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)
    for subdir in ['sensor_model', 'facial_model', 'future_predictor',
                   'fusion_model']:
        os.makedirs(os.path.join(config['paths']['models_output_dir'], subdir),
                    exist_ok=True)

    # ── Logging ──
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(config['paths']['logs_dir'], 'training.log'),
                mode='a',
                encoding='utf-8',
            ),
        ]
    )
    logger = logging.getLogger(__name__)

    # ── Seeds & Device ──
    seed = config['training'].get('seed', 42)
    set_seeds(seed)
    device = get_device(args, config)

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║" + " 🧠 MINDSENSE AI — TRAINING PIPELINE ".center(58) + "║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info(f"  Device:  {device}")
    logger.info(f"  Seed:    {seed}")
    logger.info(f"  Config:  {args.config}")
    logger.info(f"  Models:  {config['paths']['models_output_dir']}")
    if torch.cuda.is_available():
        logger.info(f"  GPU:     {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM:    {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    total_start = time.time()
    sensor_data = None
    xgb_model = None

    if not args.eval_only:
        # ════════════════════════════════════════════════════════════
        # STEP 1: Train Sensor Model (Ensemble)
        # ════════════════════════════════════════════════════════════
        logger.info("\n" + "━" * 60)
        logger.info("STEP 1/5: Training Sensor Ensemble …")
        logger.info("━" * 60)
        step_start = time.time()

        from training.train_sensor_model import train_sensor_model
        ensemble, sensor_data = train_sensor_model(config, device)
        xgb_model = ensemble.xgb_model

        elapsed = time.time() - step_start
        logger.info(f"  ✅ Sensor model complete ({format_time(elapsed)})")

        # ════════════════════════════════════════════════════════════
        # STEP 2: Train Facial Model
        # ════════════════════════════════════════════════════════════
        if not args.skip_facial:
            logger.info("\n" + "━" * 60)
            logger.info("STEP 2/5: Training Facial Emotion Model …")
            logger.info("━" * 60)
            step_start = time.time()

            from training.train_facial_model import train_facial_model
            facial_model, class_names, history = train_facial_model(
                config, device
            )

            elapsed = time.time() - step_start
            logger.info(f"  ✅ Facial model complete ({format_time(elapsed)})")
        else:
            logger.info("\n  ⏭️  Skipping facial model (--skip-facial)")

        # ════════════════════════════════════════════════════════════
        # STEP 3: Train Future Predictor
        # ════════════════════════════════════════════════════════════
        logger.info("\n" + "━" * 60)
        logger.info("STEP 3/5: Training Future State Predictor …")
        logger.info("━" * 60)
        step_start = time.time()

        from training.train_future_predictor import train_future_predictor
        future_model = train_future_predictor(
            config, device, sensor_data=sensor_data
        )

        elapsed = time.time() - step_start
        logger.info(f"  ✅ Future predictor complete ({format_time(elapsed)})")

        # ════════════════════════════════════════════════════════════
        # STEP 4: Train Fusion Model
        # ════════════════════════════════════════════════════════════
        if not args.skip_fusion:
            logger.info("\n" + "━" * 60)
            logger.info("STEP 4/5: Training Fusion Model …")
            logger.info("━" * 60)
            step_start = time.time()

            from training.train_fusion_model import train_fusion_model
            fusion_model = train_fusion_model(
                config, device,
                xgb_model=xgb_model,
                sensor_data=sensor_data,
            )

            elapsed = time.time() - step_start
            logger.info(f"  ✅ Fusion model complete ({format_time(elapsed)})")
        else:
            logger.info("\n  ⏭️  Skipping fusion model (--skip-fusion)")

    # ════════════════════════════════════════════════════════════
    # STEP 5: Evaluate All
    # ════════════════════════════════════════════════════════════
    logger.info("\n" + "━" * 60)
    logger.info("STEP 5/5: Running Full Evaluation …")
    logger.info("━" * 60)
    step_start = time.time()

    from evaluate.evaluate_all import evaluate_all
    sensor_metrics, facial_metrics, future_metrics = evaluate_all(
        config, device, sensor_data=sensor_data
    )

    elapsed = time.time() - step_start
    logger.info(f"  ✅ Evaluation complete ({format_time(elapsed)})")

    # ════════════════════════════════════════════════════════════
    # DONE
    # ════════════════════════════════════════════════════════════
    total_elapsed = time.time() - total_start
    models_dir = config['paths']['models_output_dir']

    logger.info("\n" + "═" * 60)
    logger.info("✅ All models trained. Models saved to: "
                 f"{os.path.abspath(models_dir)}")
    logger.info(f"   Total time: {format_time(total_elapsed)}")
    logger.info("═" * 60)

    # List saved artifacts
    logger.info("\n  📂 Saved artifacts:")
    for root, dirs, files in os.walk(models_dir):
        level = root.replace(models_dir, '').count(os.sep)
        indent = '  ' + '│   ' * level
        logger.info(f"{indent}├── {os.path.basename(root)}/")
        sub_indent = '  ' + '│   ' * (level + 1)
        for file in sorted(files):
            size = os.path.getsize(os.path.join(root, file))
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            logger.info(f"{sub_indent}├── {file} ({size_str})")


if __name__ == '__main__':
    main()
