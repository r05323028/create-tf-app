# Paths
IGNORE_PATHS = [
    '.vscode',
    'DS_Store',
    '__pycache__',
    'logs',
    '.pytest_cache',
    '!.gitkeep'
]

ROOT_DIRS = [
    'datasets', 
    'models',
    '.vscode'
]

NOT_INSERT_KEEP = [
    '.vscode'
]

# Requirements
REQ_PACKAGES = [
    'tensorflow==1.13.1',
    'pytest==4.3.0',
    'scikit-learn==0.20.2',
]

# Environments
DEFAULT_PYTHON_INTERPRETER = '/usr/bin/python3'
DEFAULT_UNITTEST_MODULE = 'pytest'