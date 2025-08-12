"""
BioHama 설치 스크립트

BioHama: 바이오-인스파이어드 하이브리드 적응형 메타 아키텍처
"""

from setuptools import setup, find_packages
import os

# README 파일 읽기
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "BioHama: 바이오-인스파이어드 하이브리드 적응형 메타 아키텍처"

# requirements.txt 파일 읽기
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="biohama",
    version="0.1.0",
    author="BioHama Development Team",
    author_email="contact@biohama.ai",
    description="바이오-인스파이어드 하이브리드 적응형 메타 아키텍처",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/biohama",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/biohama/issues",
        "Documentation": "https://biohama.readthedocs.io",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "full": [
            "transformers>=4.20.0",
            "opencv-python>=4.5.0",
            "librosa>=0.9.0",
            "gym>=0.21.0",
            "fastapi>=0.95.0",
            "uvicorn>=0.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "biohama=biohama.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "biohama": ["configs/*.yaml", "configs/*.yml"],
    },
    zip_safe=False,
    keywords=[
        "artificial intelligence",
        "machine learning",
        "neuroscience",
        "cognitive architecture",
        "reinforcement learning",
        "meta learning",
        "brain-inspired",
        "adaptive systems",
    ],
)
