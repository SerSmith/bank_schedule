import os
from pathlib import Path
from setuptools import (
    find_packages,
    setup,
)

src_dir = os.path.join( Path(__file__).parent, 'src' )

setup(
    name='bank_schedule',
    version='0.0.1',
    author='optimists',
    author_email='',
    description='Сервис по построению оптимальных маршрутов инкассации платежных терминалов',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    package_data={
        },
    include_package_data=True,
    extras_require={
        'test': [
            'pycodestyle',
            'pylint',
            'pylint-quotes',
            'pytest',
            'pytest-cov',
            'pytest-mock',
            'pytest-pythonpath',
            'bandit',
            'pylint'
        ],
    }
)