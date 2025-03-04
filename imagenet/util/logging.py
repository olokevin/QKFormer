import os
import sys
import yaml
from loguru import logger as _logger


__all__ = ['logger']

_logger.remove()
_logger.add(sys.stdout,
            level='DEBUG',
            format=(
                # '<green>[{time:YYYY-MM-DD HH:mm:ss.SSS}]</green> '
                '<level>{message}</level>')
            )

def configs2dict(cfg):
    from easydict import EasyDict
    if isinstance(cfg, EasyDict):
        cfg = dict(cfg)
        key2cast = [k for k in cfg if isinstance(cfg[k], EasyDict)]
        for k in key2cast:
            cfg[k] = configs2dict(cfg[k])
        return cfg
    else:
        return cfg

class ExpLogger:
    def init(self, configs):
        assert configs.run_dir is not None, 'Empty run directory!'
        
        # dumping running configs
        _path = os.path.join(configs.run_dir, 'config.yaml')
        with open(_path, 'w') as f:
            yaml.dump(configs2dict(configs), f)

        # also dump running log to file
        _logger.add(os.path.join(configs.run_dir, 'exp.log'))

    @staticmethod
    def info(*args):
        _logger.info(*args)

    @staticmethod
    def debug(*args):
        _logger.debug(*args)


logger = ExpLogger()
