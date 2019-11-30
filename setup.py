import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kaitian-xinci",
    version="0.0.1",
    author="Huang Haiping",
    author_email="ten975118@sina.com",
    description="开天-新词，中文新词发现工具。Chinese New Word Discovery Tool。",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/denseai/kaitian-xinci",
    python_requires=">=3.6",
    packages=["kaitian"],
    package_dir={"kaitian": "kaitian"},
    extras_require={
        "tf": ["tensorflow"],
        "tfgpu": ["tensorflow-gpu"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
)