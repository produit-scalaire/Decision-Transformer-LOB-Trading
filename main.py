import os
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
import torch

# Assuming your existing files are wrapped in functions instead of __main__ blocks.
# Example: def run_generation(cfg: DictConfig): ...
# You will need to slightly adapt your 3 scripts to accept this cfg object.
from src.data.trajectories_generator import generate_dataset_pipeline
from src.training.training_pipeline import train_model
from src.evaluations.dt_viz import evaluate_model

# Initialize logger
log = logging.getLogger(__name__)

def setup_environment(cfg: DictConfig) -> None:
    """
    Ensure all required directories exist and set environment variables.
    """
    Path(cfg.paths.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.plot_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.paths.train_data).parent.mkdir(parents=True, exist_ok=True)
    
    # Threading optimizations for AMD Ryzen 9 9950X
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    # Matrix multiplication precision for Ampere/Ada/Blackwell architectures
    torch.set_float32_matmul_precision('high')


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the Decision Transformer pipeline.
    Hydra automatically injects the DictConfig object.
    """
    log.info(f"Pipeline initialized with configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    setup_environment(cfg)

    # ---------------------------------------------------------
    # 1. Data Generation Phase
    # ---------------------------------------------------------
    if cfg.pipeline.run_generation:
        log.info("Starting Data Generation Phase...")
        # Inject specific configs downstream
        reward_shaping = OmegaConf.to_container(
            cfg.generator.reward_shaping, resolve=True
        )
        # reward_horizon is None (full episode) or a positive integer
        reward_horizon = cfg.generator.get("reward_horizon", None)
        generate_dataset_pipeline(
            train_episodes=cfg.generator.train_episodes,
            test_episodes=cfg.generator.test_episodes,
            workers=cfg.hardware.workers,
            train_out=cfg.paths.train_data,
            test_out=cfg.paths.test_data,
            plot_dir=cfg.paths.plot_dir,
            window_size=cfg.generator.window_size,
            episode_length=cfg.generator.episode_length,
            reward_type=cfg.generator.reward_type,
            reward_shaping=reward_shaping,
            state_representation=cfg.generator.state_representation,
            price_offset=float(cfg.generator.price_offset),
            reward_horizon=int(reward_horizon) if reward_horizon is not None else None,
        )
    else:
        log.info("Skipping Data Generation Phase.")

    # ---------------------------------------------------------
    # 2. Training Phase
    # ---------------------------------------------------------
    if cfg.pipeline.run_training:
        log.info("Starting Training Phase...")
        # Pass the full sub-configs for model and training
        train_model(
            train_data_path=cfg.paths.train_data,
            model_dir=cfg.paths.model_dir,
            model_cfg=cfg.model,
            train_cfg=cfg.training,
            hardware_cfg=cfg.hardware,
            plot_dir=cfg.paths.plot_dir,
        )
    else:
        log.info("Skipping Training Phase.")

    # ---------------------------------------------------------
    # 3. Evaluation & Visualization Phase
    # ---------------------------------------------------------
    if cfg.pipeline.run_evaluation:
        log.info("Starting Evaluation Phase...")
        # Evaluate the latest saved checkpoint
        latest_model = f"{cfg.paths.model_dir}/dt_model_final.pt"
        reward_shaping_eval = OmegaConf.to_container(
            cfg.generator.reward_shaping, resolve=True
        )
        evaluate_model(
            model_path=latest_model,
            data_path=cfg.paths.test_data,
            eval_cfg=cfg.evaluation,
            model_cfg=cfg.model,
            plot_dir=cfg.paths.plot_dir,
            state_representation=cfg.generator.state_representation,
            train_data_path=cfg.paths.train_data,
            generator_window_size=int(cfg.generator.window_size),
            generator_price_offset=float(cfg.generator.price_offset),
            generator_reward_type=str(cfg.generator.reward_type),
            generator_reward_shaping=reward_shaping_eval,
        )
    else:
        log.info("Skipping Evaluation Phase.")

    log.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()