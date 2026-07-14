#!/usr/bin/env python3
"""IF-quantify-3channels: 免疫荧光三通道定量分析工具"""
from setuptools import setup

setup(
    name="if-quantify",
    version="2.0.0",
    description="免疫荧光三通道定量分析 - CD138分选阳性细胞全场比值法",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Yibo Zhang",
    url="https://github.com/yibogogogo/IF-quantify-3channels-",
    py_modules=["if_quantify"],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "Pillow>=9.0",
        "scipy>=1.7",
        "scikit-image>=0.19",
    ],
    extras_require={
        "stardist": ["stardist", "tensorflow"],
        "cellpose": ["cellpose", "torch"],
    },
    entry_points={
        "console_scripts": [
            "if-quantify=if_quantify:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
