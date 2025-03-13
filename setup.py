#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="evil2root-trading",
    version="1.0.0",
    author="EVIL2ROOT Team",
    author_email="contact@evil2root.ai",
    description="Un système de trading automatisé avec validation IA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EpicSanDev/EVIL2ROOT_AI-main",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "evil2root-trading=src.main:main",
            "evil2root-backtest=src.core.backtesting:main",
            "evil2root-analyze=src.models.ensemble.ensemble_model:main",
        ],
    },
    include_package_data=True,
) 