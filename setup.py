from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("NPTFit_ftreg.npll", ["NPTFit_ftreg/npll.pyx"],
        include_dirs=[numpy.get_include()], extra_compile_args=["-ffast-math",'-O3']),
    Extension("NPTFit_ftreg.pll", ["NPTFit_ftreg/pll.pyx"],
        include_dirs=[numpy.get_include()], extra_compile_args=["-ffast-math",'-O3']),
    Extension("NPTFit_ftreg.incgamma_fct_p", ["NPTFit_ftreg/incgamma_fct_p.pyx"],
        include_dirs=[numpy.get_include()], extra_compile_args=["-ffast-math",'-O3']),
    Extension("NPTFit_ftreg.x_m", ["NPTFit_ftreg/x_m.pyx"],
        include_dirs=[numpy.get_include()], extra_compile_args=["-ffast-math",'-O3']),
    Extension("NPTFit_ftreg.incgamma_fct", ["NPTFit_ftreg/incgamma_fct.pyx"],
        include_dirs=[numpy.get_include()], libraries=["gsl", "gslcblas", "m"],
        extra_compile_args=["-ffast-math",'-O3']),
    Extension("NPTFit_ftreg.interp1d", ["NPTFit_ftreg/interp1d.pyx"],
        include_dirs=[numpy.get_include()], extra_compile_args=["-ffast-math",'-O3']),
    Extension("NPTFit_ftreg.findmin", ["NPTFit_ftreg/findmin.pyx"],
        include_dirs=[numpy.get_include()], extra_compile_args=["-ffast-math",'-O3'])
]

setup(
    name='NPTFit_ftreg',
    version='0.1.1',
    description='A Python package for Non-Poissonian Template Fitting',
    url='https://github.com/ljchang/NPTFit',
    author='Siddharth Mishra-Sharma',
    author_email='smsharma@princeton.edu',
    license='MIT',
    install_requires=[
            'numpy',
            'matplotlib',
            'healpy',
            'Cython',
            'pymultinest',
            'jupyter',
            'corner',
            'mpmath',
        ],

    packages=['NPTFit_ftreg'],
    ext_modules = cythonize(extensions)
)
