import pdb

from rich.console import Console
from rich.table import Table
from rich.pretty import Pretty
from rich.progress import Progress
import yaml
import inspect
import os
import shutil
import glob

def case_insensitive_glob(pathname):
    # Split path into directory and pattern
    dirname, pattern = os.path.split(pathname)
    return [os.path.join(dirname, f) for f in os.listdir(dirname)
            if glob.fnmatch.fnmatchcase(f.lower(), pattern.lower())]

def copy_file_with_progress(src_path, dst_path, progress=None):
    """
    Copy a file from src_path to dst_path with a progress bar.

    Parameters
    ----------
    src_path : str
        Path to the source file.
    dst_path : str
        Path to the destination file.
    """
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"Source file not found: {src_path}")

    desc = f"Copying {os.path.basename(src_path)}"
    with progress.open(src_path, "rb", description=desc) as src:
        with open(dst_path, "wb") as dst:
            shutil.copyfileobj(src, dst)


def read_yaml(yaml_path):

    # Read YAML file
    with open(yaml_path, 'r') as stream:
        yaml_dict = yaml.safe_load(stream)

    return yaml_dict

def read_params(yaml_path, process_name=None):

    yaml_dict = read_yaml(yaml_path)

    if process_name is None:
        processes = list(yaml_dict.keys())
        params = yaml_dict[processes[0]]
        params['process_name'] = processes[0]
    else:
        params = yaml_dict[process_name]
        params['process_name'] = process_name


    return params

def print_params(params):
    console = Console()
    table = Table('Parameter', 'Value')
    for key, value in params.items():

        if type(value) is list:
            if type(value[0]) is str:
                value = ', '.join(value)
            else:
                value = str(value)
        elif (type(value) is int) or (type(value) is float):
            value = str(value)
        elif type(value) is dict:
            value = Pretty(value)

        table.add_row(key, value)

    console.print(table)

def params_to_mpl_metadata(params):
    """
    Convert params values to a format that is compatible with saving as a metadata in matplotlib
    :param params:
    :return:
    """
    metadata = {}

    for key, value in params.items():

        if type(value) is list:
            value = ', '.join(value)

        metadata[key] = value

    return metadata


def filter_params(params, func):

    filtered_params = {k: v for k, v in params.items() if
                       k in [p.name for p in inspect.signature(func).parameters.values()]}

    return filtered_params