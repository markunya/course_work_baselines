from omegaconf import OmegaConf
from utils.model_utils import setup_seed
from utils.data_utils import load_config
from training.trainers.base_trainer import gan_trainers_registry

if __name__ == "__main__":
    config = load_config()
    conf_cli = OmegaConf.from_cli()
    run_name = conf_cli.exp.run_name
    config = OmegaConf.merge(config, conf_cli)
    setup_seed(config.exp.seed)

    trainer_args = {}
    if 'trainer_args' in config.train:
        trainer_args = config.train.trainer_args
    trainer = gan_trainers_registry[config.train.trainer](config, **trainer_args)

    trainer.setup_training()
    trainer.training_loop()