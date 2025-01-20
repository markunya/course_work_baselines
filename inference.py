from utils.model_utils import setup_seed
from utils.model_utils import load_config

if __name__ == "__main__":
    config = load_config()
    setup_seed(config.exp.seed)

    if config.exp.model_type == "gan":
        from training.trainers.hifigan_trainer import gan_trainers_registry
        trainer = gan_trainers_registry[config.train.trainer](config)
    else:
        raise ValueError(f"Unknown model type {config.exp.model_type}")

    trainer.setup_inference()
    trainer.inference()