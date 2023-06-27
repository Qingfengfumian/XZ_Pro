import yaml
from attrdict import AttrDict
import argparse
import os
import inspect
import warnings
warnings.filterwarnings("ignore")

current_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
config_path = os.path.join(current_path, 'config')
config_file = os.path.join(config_path, 'config.yaml')


def read_yaml(filepath):
    # with open(filepath, encoding="utf-8") as f:
    with open(filepath, encoding="utf-8") as f:
        config = yaml.load(f)
    return AttrDict(config)


class Params_con(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--cmd', type=str, default="")
        # self.parser.add_argument('--mode', type=str, default="")
        self.get_config()

    def get_config(self):
        config = read_yaml(config_file)
        for conf in config.keys():
            self.parser.add_argument('--' + conf, type=str, default=config[conf])

    def params(self):
        param = self.parser.parse_args()
        return param


param_con = Params_con()
