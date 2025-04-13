from setuptools import setup, find_packages


# 自动读取 requirements.txt
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="your-package-name",    # 包名称（PyPI 显示的名称）
    version="0.1.0",            # 版本号（推荐语义化版本）
    author="Hengjia Xiao",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/speedhawk/LLM-A-",
    packages=find_packages(),   # 自动发现所有 Python 包
    install_requires=required,  # 直接使用 requirements.txt 内容
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',    # 指定 Python 版本要求
)