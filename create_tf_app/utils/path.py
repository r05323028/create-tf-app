import re
import os
from jinja2 import Environment, PackageLoader
from create_tf_app.settings import (
    IGNORE_PATHS, REQ_PACKAGES, DEFAULT_PYTHON_INTERPRETER, ROOT_DIRS, NOT_INSERT_KEEP)


def create_empty_file(dir, fname, **kwargs):
    init_fp = os.path.join(dir, fname)

    if not os.path.exists(init_fp):
        with open(init_fp, 'w') as init_file:
            init_file.close()

def create_root_dir(root_dir, **kwargs):    
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

def create_root_dirs(root_dir, app_name, **kwargs):
    dirs = [app_name] + ROOT_DIRS

    for name in dirs:
        path = os.path.join(root_dir, name)

        if not os.path.exists(path):
            os.mkdir(path)

            if name not in [app_name] + NOT_INSERT_KEEP:
                insert_git_keep(path)

def insert_git_keep(dir):
    path = os.path.join(dir, '.gitkeep')

    if not os.path.exists(path):
        create_empty_file(dir, '.gitkeep')

def build_app(root_dir, app_name, **kwargs):
    app_path = os.path.join(root_dir, app_name)
    layers_path = os.path.join(app_path, 'layers')

    if not os.path.exists(layers_path):
        os.mkdir(layers_path)

    create_empty_file(layers_path, '__init__.py')

def build_estimator(root_dir, app_name, **kwargs):
    app_path = os.path.join(root_dir, app_name)
    estimator_path = os.path.join(app_path, '__init__.py')

    if not os.path.exists(estimator_path):
        with open(estimator_path, 'w') as file:
            env = Environment(loader=PackageLoader('create_tf_app', 'templates'))

            if kwargs.get('use_example') == 'iris':
                template = env.get_template('estimators/examples/iris.j2')
                output = template.render()
            else:
                template = env.get_template('estimators/examples/empty.j2')
                output = template.render()
            file.write(output)

def build_model_scripts(root_dir, app_name, **kwargs):
    train_py_path = os.path.join(root_dir, 'train.py')

    if not os.path.exists(train_py_path):
        env = Environment(loader=PackageLoader('create_tf_app', 'templates'))

        if kwargs.get('use_example') == 'iris':
            template = env.get_template('scripts/examples/iris.j2')
            output = template.render(app_name=app_name)

        else:
            template = env.get_template('scripts/examples/empty.j2')
            output = template.render(app_name=app_name)

        with open(train_py_path, 'w') as file:
            file.write(output)

def build_tests(root_dir, **kwargs):
    test_path = os.path.join(root_dir, 'tests')

    if not os.path.exists(test_path):
        os.mkdir(test_path)
        create_empty_file(test_path, '__init__.py')

def make_app_config_files(root_dir, **kwargs):
    ignore_fp = os.path.join(root_dir, '.gitignore')
    req_fp = os.path.join(root_dir, 'requirements.txt')
    readme_fp = os.path.join(root_dir, 'README.md')

    vscode_settings = os.path.join(root_dir, '.vscode', 'settings.json')

    if not os.path.exists(ignore_fp):
        with open(ignore_fp, 'w') as ignore_file:
            ignore_file.writelines([line + '\n' for line in IGNORE_PATHS])

    if not os.path.exists(req_fp):
        with open(req_fp, 'w') as req_file:
            for line in REQ_PACKAGES:
                if kwargs.get('use_gpu'):
                    line = line.replace('tensorflow', 'tensorflow-gpu')
                    req_file.write(line + '\n')

                if not kwargs.get('use_example') in ['iris']:
                    if re.match('scikit-learn.*', line):
                        continue
                    
                req_file.write(line + '\n')

    if not os.path.exists(readme_fp):
        with open(readme_fp, 'w') as readme_file:
            readme_file.close()

    if not os.path.exists(vscode_settings):
        env = Environment(loader=PackageLoader('create_tf_app', 'templates'))
        template = env.get_template('misc/vscode.j2')
        output = template.render(python_interpreter=kwargs.get('python_interpreter', DEFAULT_PYTHON_INTERPRETER))

        with open(vscode_settings, 'w') as settings:
            settings.write(output)