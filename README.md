C-CONCOCT
=========

Multithreaded C implementation of a Gaussian mixture model designed to speed up the clustering stage of CONCOCT but applicable to any GMM clustering.

The code heavily exploits the [Gnu Science Library] (http://www.gnu.org/software/gsl/). This must be pre-installed but otherwise it should be installable on any system with a C compiler although here we are assuming gcc. To compile:

    make

This will generate the executable EMGMM which should then be added to your bin directory or elsewhere in your path. Performance may be improved by changing the compiler options or linking to an optimised BLAS library although we do not recommend OpenBLAS because the code is using pthreads for optimisation already. This version is hardcoded to use 40 threads. 

 
