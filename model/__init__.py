from .graspcvae import GraspCVAE
def build_model(cfg):
    if cfg.name.lower() == "graspcvae":
        return GraspCVAE(cfg)
    else:
        raise NotImplementedError(f"No such model: {cfg.name}")
