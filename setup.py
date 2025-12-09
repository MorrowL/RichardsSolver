from distutils.core import setup
import sys

script_args = sys.argv[1:]

setup(name='RichardsSolver',
      version='0.10',
      author='Liam Morrow',
      author_email='liam.morrow@anu.edu.au',
      description='A numerical package for solving Richards equation',
      long_description=open('README.rst').read(),
      url='https://github.com/g-adopt/RichardsSolver',
      packages=['RichardsSolver'],
      install_requires=['numpy', 'scipy'],
      keywords=['Richards equation', 'analytical solutions',
                'groundwater flow', 'unsaturated flow', 'benchmarks'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: GNU Lesser General Public License v3 (LGPLv3)',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering'],
      script_args=script_args,
      ext_package='RichardsSolver',
      )
