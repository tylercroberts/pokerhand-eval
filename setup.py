# -*- coding: utf-8 -*-
from setuptools import setup

# get the dependencies and installs
with open("requirements.txt", "r", encoding="utf-8") as f:
    requires = [x.strip() for x in f if x.strip()]

setup(
    name='pokereval',
    version='0.3.0',
    author='Alvin Liang, fork by Tyler Roberts',
    author_email='tcroberts@live.ca',
    packages=['pokereval'],
    url='https://github.com/tylercroberts/pokerhand-eval',
    license='Apache, see LICENSE.txt',
    description='A pure python poker hand evaluator for 5, 6, 7 cards',
    long_description=open('README.md').read(),
    install_requires=requires,
    extras_require={
        "docs": [
            "sphinx>=1.6.3, <2.0",
            "sphinx_rtd_theme==0.4.1",
            "nbsphinx==0.3.4",
            "nbstripout==0.3.3",
            "recommonmark==0.5.0",
            "sphinx-autodoc-typehints==1.6.0",
            "sphinx_copybutton==0.2.5",
        ],
        "tests": ["pytest==5.2.1",
                  "pytest-cov==2.8.1",
                  "coverage==3.7.1",
                  "python-coveralls==2.4.3"]
    },
)
