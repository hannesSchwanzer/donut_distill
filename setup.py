from setuptools import setup, find_packages

setup(
    name="donut_distill",
    version="0.1.0",
    author="Hannes Schwanzer",
    author_email="hannes.schwanzer@gmail.com",
    url="https://github.com/hannesSchwanzer/donut_funsd",
    packages=find_packages(),
    install_requires=[
        "anls",
        "datasets",
        "matplotlib",
        "nltk",
        "numpy",
        "Pillow",
        "Pillow",
        "pytorch_lightning",
        "PyYAML",
        "PyYAML",
        "sentencepiece",
        "setuptools",
        "torch",
        "torchvision",
        "tqdm",
        "transformers",
        "wandb",
    ],
    python_requires=">=3.11.2",
)

