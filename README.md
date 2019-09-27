# ptacake: Pulsar Timing Arrays - Correlations Are KEy
Python package for PTA data analysis based on correlations between the pulsars: nullstreams and spherical harmonics (paper references?). The package also includes tools to simulate PTA data with realistic sampling (uneven time stamps, different TOAs for each pulsar, gaps, ...).

# install locally
You can install ptacake locally. To do this, first clone the git repository with ssh (recommended):
```bash
git clone git@github.com:cjm96/NullStreams.git
```
or with https:
```bash
git clone https://github.com/cjm96/NullStreams.git
```
Then go into the repository directory, and use pip to install:
```bash
pip install .
```
you can add the option -e (after pip install) to install in development mode, which makes it so that the install is affected by any changes made to the source code. Or if you want to specify the install directory:
```bash
python setup.py install --prefix=/path/to/directory
```

# install in a virtual environment
I recommend using a virtual environment to install the package (especially if you can use this on a cluster). To do this, first make a virtual environment:
```bash
virtualenv venv
```
venv can be anything, it's just a name for your virtual environment. This way, virtualenv uses whichever is the default python on your system. Use python 3.6 for ptacake. For more details (e.g. how to specify the python version) check https://www.alexkras.com/how-to-use-virtualenv-in-python-to-install-packages-locally/.
Now, we go activate the virtual environment with:
```bash
source venv/bin/activate
```
Then, install the python modules needed for ptacake, which are written in requirements.txt (in the repository). After that, install ptacake itself:
```bash
pip install -r requirement.txt
pip install -e .
```
(-e again optional for development mode). You can now use ptacake. Check you can run the unittest with:
```bash
python -m unittest discover tests
```
If you want to quit the virtual environment, run
```bash
deactivate
```

# testing
Run unittests from the git repository with:
```bash
python -m unittest discover tests
```
I am assuming here that python defaults to python3 (check with ```which python```). Note that at least one test fails with python2 (due to a change with using * to unpack stuff). New test modules should follow the naming convention `test_<something_descriptive>.py`.


