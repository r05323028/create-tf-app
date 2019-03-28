from setuptools import setup, find_packages


entry_points = '''
[console_scripts]
create-tf-app=create_tf_app.bin.cli:cli
'''

setup(
    name='create_tf_app',
    version='0.0.3',
    install_requires=[
        'Jinja2==2.10',
        'Click==7.0',
        'tqdm==4.23.4'
    ],
    description='Command line tool to create tensorflow project based on the high-level api.',
    url='https://github.com/r05323028/create-tf-app',
    include_package_data=True,
    packages=find_packages(include=['create_tf_app.*', '*']),
    author='seanchang',
    author_email='seanchang@kkbox.com',
    entry_points=entry_points
)