import torch
from models.tacotron import Tacotron

def get_distributed_tacotron(tacotron: Tacotron):
    # distr_model = torch.nn.parallel.DistributedDataParallel(tacotron)
    distr_model = torch.nn.DataParallel(tacotron)
    distr_model.r = distr_model.module.r
    distr_model.generate = distr_model.module.generate
    distr_model.init_model = distr_model.module.init_model
    distr_model.get_step = distr_model.module.get_step
    distr_model.reset_step = distr_model.module.reset_step
    distr_model.log = distr_model.module.log
    distr_model.load = distr_model.module.load
    distr_model.save = distr_model.module.save
    distr_model.num_params = distr_model.module.num_params
    return distr_model