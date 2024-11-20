from setuptools import setup
from setuptools.discovery import PackageFinder

setup(name='LiLab',
            version='0.1',
            author="Naiqi Li",
            license="Apache License 2.0",
            packages=PackageFinder.find(),
            classifiers=['License :: OSI Approved :: Apache Software License']
      )

