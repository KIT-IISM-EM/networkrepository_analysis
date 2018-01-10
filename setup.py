from distutils.core import setup

setup(
    name='networkrepository_analysis',
    version='0.1.1',
    packages=[''],
    url='https://github.com/FabianBall/networkrepository_analysis',
    license='MIT',
    author='Fabian Ball',
    author_email='fabian.ball@kit.edu',
    description='Several analysis scripts that were used to analyze data from networkrepository.com '
                'for graph automorphisms',
    requires=['beautifulsoup4',
              'future',
              'matplotlib',
              'networkx',
              'numpy',
              'pandas',
              'requests',
              'scipy',

              'pycggcrg',
              'pysaucy'],
)
