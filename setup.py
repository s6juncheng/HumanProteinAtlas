#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "scikit-image",
    "sklearn"
]

test_requirements = [
    "pytest",
    "virtualenv",
]

setup(
    name='hpa_src',
    version='0.0.1',
    description='Human Protein Atlas Kaggle challenge',
    author='Jun Cheng',
    author_email='...',
    url='https://github.com/s6juncheng/HumanProteinAtlas',
    long_description=readme,
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "develop": ["bumpversion",
                    "wheel",
                    "jedi",
                    "epc",
                    "pytest",
                    "pytest-pep8",
                    "pytest-cov"],
    },
    entry_points={'console_scripts': ['hpa_src = hpa_src.__main__:main']},
    license="MIT license",
    zip_safe=False,
    test_suite='tests',
    package_data={},
    tests_require=test_requirements
)