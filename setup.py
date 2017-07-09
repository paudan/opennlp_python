import os
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.rst')).read()

setup(name='nltk-opennlp',
      version='1.0.2',
      description='NLTK interface with OpenNLP',
      long_description=README,
      author='Paulius Danenas',
      author_email='danpaulius@gmail.com',
      url='https://github.com/paudan/opennlp-python',
      py_modules=['nltk-opennlp'],
      packages=['nltk_opennlp'],
      install_requires=['nltk'],
      license='GPL Version 3',
    )



