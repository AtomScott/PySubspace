from setuptools import setup, find_packages

setup(
    name='cvt',
    version='0.1.0',
    packages=['cvt'],
    install_requires=[
        'numpy',
        'scikit-learn',
        'numba'
    ]
)
