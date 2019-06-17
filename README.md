# ptacake: Pulsar Timing Arrays: Correlations Are KEy
package for simulating realistically sampled (uneven sampling, different per pulsar, gaps, ...) PTA data and analysis based on null streams or on spherical harmonics.
[paper references?]

# build
The package is made based on instructions from https://dzone.com/articles/executable-package-pip-install
But I added some stuff after reading https://packaging.python.org/tutorials/packaging-projects/ since we have a full python package to install, not just one executable.
After cloning the repository, go the repository directory, then build with:
`
python setup.py sdist bdist_wheel
`

# install
After building, you can pip install the package locally. Go to the git repository, then run:
`
python -m pip install dist/ptacake<stuff>.whl
`
where the file with `<stuff>` in the name gets made automatically after the build; there should only be one .whl file to install.

