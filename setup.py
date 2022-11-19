import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sphaera",
    version="0.0.1",
    author="Mingli Yuan",
    author_email="mingli.yuan@gmail.com",
    description="sphaera is a math toolkit for spherical data processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mountain/sphaera",
    project_urls={
        'Documentation': 'https://github.com/mountain/sphaera',
        'Source': 'https://github.com/mountain/sphaera',
        'Tracker': 'https://github.com/mountain/sphaera/issues',
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'cached_property',
        'numpy',
        'scipy',
        'torch',
        'vtk',
        'pyvista',
        'netcdf4',
        'h5netcdf',
        'xarray',
    ],
    test_suite='nose.collector',
    tests_require=['pytest'],
)

