import yaml
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp


def load_hparam_space():
    with open('configs/all_hparams.yaml') as f:
        hparams_dict = yaml.full_load(f)

        hp_space = dict()

        for name, values in hparams_dict.items():
            hp_space[name] = hp.HParam(name, hp.Discrete(values))

    return hp_space


def parse_hparams(yaml_file, hp_space):
    with open(yaml_file) as f:
        hparams_dict = yaml.full_load(f)

    hparams = dict()

    for name, val in hparams_dict.items():
        hparams[hp_space[name]] = val

    return hparams


if __name__ == '__main__':
    hp_space = load_hparam_space()
    hparams = parse_hparams('src/config/128_segmentation-no-reg-lr-005-region.yaml', hp_space)
    train_val_region_split = hparams[hp_space['train_val_region_split']]
    print(type(train_val_region_split))
