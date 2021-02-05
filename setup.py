import sys

from setuptools import setup, find_packages

import nvhtop


setup(
    name='nvhtop',
    version=nvhtop.__version__,
    description=nvhtop.__doc__,
    license=nvhtop.__license__,
    author=nvhtop.__author__,
    author_email=nvhtop.__email__,
    url="https://github.com/XuehaiPan/nvhtop.git",
    packages=find_packages(include=['nvhtop', 'nvhtop.*']),
    entry_points={
        'console_scripts': [
            'nvhtop=nvhtop:main',
        ],
    },
    install_requires=[
        'nvidia-ml-py',
        'psutil',
        'cachetools',
        'termcolor'
    ] + (['windows-curses'] if sys.platform == 'windows' else []),
    python_requires='>=3.5',
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Environment :: Console :: Curses',
        'Intended Audience :: Developers',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: System Administrators',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Topic :: System :: Hardware',
        'Topic :: System :: Systems Administration',
        'Topic :: Utilities',
    ],
)
