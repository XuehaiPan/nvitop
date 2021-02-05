import sys

from setuptools import setup, find_packages

import nvhtop


extra_description = {}
try:
    with open('README.md', mode='r') as doc:
        extra_description['long_description'] = doc.read()
        extra_description['long_description_content_type'] = 'text/markdown'
except OSError:
    pass

setup(
    name='nvhtop',
    version=nvhtop.__version__,
    description=nvhtop.__doc__,
    **extra_description,
    license=nvhtop.__license__,
    author=nvhtop.__author__,
    author_email=nvhtop.__email__,
    url="https://github.com/XuehaiPan/nvhtop.git",
    packages=find_packages(include=['nvhtop', 'nvhtop.*']),
    entry_points={
        'console_scripts': [
            'nvhtop=nvhtop.main:main'
        ],
    },
    install_requires=(['windows-curses'] if sys.platform == 'windows' else []) + [
        'nvidia-ml-py',
        'psutil',
        'cachetools',
        'termcolor'
    ],
    python_requires='>=3.5, <4',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Environment :: GPU',
        'Environment :: Console',
        'Environment :: Console :: Curses',
        'Intended Audience :: Developers',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: System Administrators',
        'Topic :: System :: Hardware',
        'Topic :: System :: Monitoring',
        'Topic :: System :: Systems Administration',
        'Topic :: Utilities'
    ],
    keywords='nvidia, nvidia-smi, GPU, htop, top',
    project_urls={
        'Bug Reports': 'https://github.com/XuehaiPan/nvhtop/issues',
        'Source': 'https://github.com/XuehaiPan/nvhtop'
    },
)
