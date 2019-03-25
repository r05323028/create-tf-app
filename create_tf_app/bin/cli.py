import os
import click
from tqdm import tqdm
from create_tf_app.settings import DEFAULT_PYTHON_INTERPRETER
from create_tf_app.utils.path import (
    create_root_dir, create_root_dirs, build_app, make_app_config_files, build_tests, build_estimator, build_model_scripts)


@click.command()
@click.option('--app_name', required=True, type=str, help='The app name.')
@click.option('--use_gpu', default=False, type=bool, help='This side project whether to use gpus.')
@click.option('--use_example', default=None, type=click.Choice(['iris']), help='The name of example project.')
@click.option('--python_interpreter', default=DEFAULT_PYTHON_INTERPRETER, help='Specify the python interpreter. (for VSCode)')
def cli(**kwargs):
    root_dir = os.path.join('./', kwargs['app_name'])
    kwargs.update({
        'root_dir': root_dir
    })

    func_registried = [
        create_root_dir, 
        create_root_dirs, 
        build_app, build_tests, 
        make_app_config_files,
        build_estimator,
        build_model_scripts
    ]

    for f in tqdm(func_registried, total=len(func_registried), desc='build_side_project'):
        f(**kwargs)
    

