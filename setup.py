import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="src",
    version="0.0.1",
    author="Louis Schlessinger",
    author_email="louschlessinger96@gmail.com",
    description="Automated model search using Bayesian optimization and genetic programming.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lschlessinger1/MS-project",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ])
