from omegaconf import DictConfig
from .BrainHGT import BrainHGT

def model_factory(config: DictConfig):

    return eval(config.model.name)(config).cuda()
