# -*- coding: utf-8 -*-
from setuptools import setup

setup(
    name='pokereval',
    version='0.3.0',
    author='Alvin Liang, fork by Tyler Roberts',
    author_email='tcroberts@live.ca',
    packages=['pokereval'],
    url='https://github.com/tylercroberts/pokerhand-eval',
    license='Apache, see LICENSE.txt',
    description='A pure python poker hand evaluator for 5, 6, 7 cards',
    long_description=open('README.rst').read(),
)
