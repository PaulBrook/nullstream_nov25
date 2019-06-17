# instructions from here https://dzone.com/articles/executable-package-pip-install
import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
            name='ptacake',
            version='0.1',
            scripts=[],
            author='Janna Goldstein, Elinore Roebber, Chris Moore',
            author_email='jgoldstein@star.sr.bham.ac.uk',
            description='PTA data analysis with null streams or spherical harmonics',
            long_description=long_description,
            long_description_content_type='text/markdown',
            url='https://github.com/cjm96/NullStreams',
            packages=setuptools.find_packages(),
            # we should add a license and add it to classifiers as
            # "License :: OSI Approved :: MIT License" for example
            classifiers=[
                "Programming Language :: Python :: 3",
                "Operating System :: OS Independent",
                ],
            )
