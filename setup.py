from setuptools import setup, find_packages

setup(
    name="vllm-hpu-extension",
    version="0.1",
    packages=find_packages(),
    install_requires=[],
    author="Tomasz Zielinski",
    author_email="tzielinski@habana.ai",
    description="HPU extension package for vLLM",
    long_description="HPU extension package for vLLM. The package contains custom HPU-specific ops. It only works together with vLLM.",
    long_description_content_type="text/markdown",
    url="https://github.com/tzielinski-habana/vllm-hpu-extension",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

