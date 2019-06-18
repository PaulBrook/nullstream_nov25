# ptacake: Pulsar Timing Arrays - Correlations Are KEy
Python package for PTA data analysis based on correlations between the pulsars: nullstreams and spherical harmonics (paper references?). The package also includes tools to simulate PTA data with realistic sampling (uneven time stamps, different TOAs for each pulsar, gaps, ...).

# build
The package is made based on instructions from https://dzone.com/articles/executable-package-pip-install
But I added some stuff after reading https://packaging.python.org/tutorials/packaging-projects/ since we have a full python package to install, not just one executable.
After cloning the repository, go the repository directory, then build with:
```bash
python3 setup.py sdist bdist_wheel
```

# install
After building, you can pip install the package locally. Go to the git repository, then run:
```bash
python3 -m pip install dist/ptacake<stuff>.whl
```
where the file with `<stuff>` in the name gets made automatically after the build; there should only be one .whl file to install.

# testing
Run unittests from the git repository with:
```bash
python3 -m unittest discover tests
```
Note that at least one test fails with python2 (due to a change with using * to unpack stuff). New test modules should follow the naming convention `test_<something_descriptive>.py`.

