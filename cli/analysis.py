import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="conf/analysis", config_name="default")
def main(cfg: DictConfig):    
    output_dir = HydraConfig.get().runtime.output_dir
    analyzer = instantiate(cfg.analyzer, _convert_="all")
    analyzer.print_datasets()
    analyzer.features_by_dataset(output_dir)


if __name__ == "__main__":
    main()
