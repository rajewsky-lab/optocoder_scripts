import yaml

def load_yaml(config_path):
    with open(config_path, 'r') as ymlfile:
        yaml_file = yaml.load(ymlfile, Loader=yaml.FullLoader)

    return yaml_file
