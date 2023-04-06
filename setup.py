from pathlib import Path

from setuptools import find_packages, setup


setup(
    name='dswx_verification',
    use_scm_version=True,
    description='Verification of DSWx Products',
    long_description=(Path(__file__).parent / 'README.md').read_text(),
    long_description_content_type='text/markdown',
    url='https://github.com/OPERA-Cal-Val/DSWx-Requirement-Verification',
    author='Alexander Handwerger, Charlie Marshak, and OPERA Project Science Team',
    author_email='opera-ps@jpl.nasa.gov',
    keywords='OPERA, Validation',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.10',
    install_requires=[
        'affine',
        'boto3',  # required for rasterio vsis3 support
        'geopandas',
        'numpy',
        'rasterio',
        'requests',
        'shapely',
        'tqdm',
        'jupytext',
        'jupyter',
        'papermill',
        'scikit-learn',
        'scikit-image',
        'numpy',
        'pandas',
    ],
    extras_require={
        'develop': [
            'flake8',
            'flake8-import-order',
            'flake8-blind-except',
            'flake8-builtins',
            'pytest',
            'pytest-cov',
        ]
    },
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
