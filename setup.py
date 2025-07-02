"""
Setup configuration for the SEM/GDS alignment tool.

This allows installation with development dependencies using:
pip install -e ".[dev]"
"""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Development dependencies
dev_requirements = [
    'pytest>=7.0.0',
    'black>=22.0.0', 
    'flake8>=5.0.0',
    'mypy>=1.0.0',
    'pytest-qt>=4.0.0',  # For Qt testing
    'pytest-cov>=4.0.0'  # For coverage
]

setup(
    name="sem-gds-comparison-tool",
    version="1.0.0",
    description="Qt6-based tool for comparing SEM images against GDSII chip layouts",
    long_description=open('README.md').read() if open('README.md') else "",
    long_description_content_type="text/markdown",
    author="Image Analysis Team",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        'dev': dev_requirements,
        'jupyter': ['jupyter>=1.0.0'],
        'analysis': ['pandas>=1.5.0', 'seaborn>=0.11.0']
    },
    entry_points={
        'console_scripts': [
            'sem-gds-tool=src.ui.main_window:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research", 
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    include_package_data=True,
    package_data={
        'src.ui.styles': ['*.qss', '*.css'],
        'src': ['*.md']
    }
)
