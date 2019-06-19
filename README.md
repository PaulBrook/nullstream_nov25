# ptacake: Pulsar Timing Arrays - Correlations Are KEy
Python package for PTA data analysis based on correlations between the pulsars: nullstreams and spherical harmonics (paper references?). The package also includes tools to simulate PTA data with realistic sampling (uneven time stamps, different TOAs for each pulsar, gaps, ...).

# install locally
Some references used to build this pacakge:
https://python-packaging.readthedocs.io/en/latest/minimal.html,
https://dzone.com/articles/executable-package-pip-install,
https://packaging.python.org/tutorials/packaging-projects

For now, you can install the package locally. Go to the git repository, then run:
```bash
pip install .
```
Add the option -e if you want the install to be affected by changes you make to the source code (so for development). 

# testing
Run unittests from the git repository with:
```bash
python3 -m unittest discover tests
```
Note that at least one test fails with python2 (due to a change with using * to unpack stuff). New test modules should follow the naming convention `test_<something_descriptive>.py`.

