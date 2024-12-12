from setuptools import setup, find_packages

setup(
    name="vllm-hpu-extension",
    packages=find_packages(),
    install_requires=[],
    long_description="HPU extension package for vLLM. The package contains custom HPU-specific ops. It only works together with vLLM.",
    long_description_content_type="text/markdown",
    url="https://github.com/HabanaAI/vllm-hpu-extension",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0",
        "Operating System :: OS Independent",
    ],
)
