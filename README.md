This is C++ based implementation for SVGPM.  

In order to build this program, please install shark machine learning library version 3 into your system https://github.com/Shark-ML/Shark/tree/3.1.1.   

This program has been sucessfully build and run on WSL2 Ubuntu 20.04 Based Distro. 

In order to run the program (once it has been built):  
        go to /build/app  
        run svgpm on command line/console:  
            svgpm  -gen    100	-p	100	-ds  wine_quality.csv.  

The program will run for 100 generation (-gen 100), with population 100 (-p 100), on dataset wine_quality.csv (-ds wine_quality.csv) using 5-fold cross-validation.

This work has been used in the following papers:

1. Pozi, M. S. M., Sulaiman, M. N., Mustapha, N., & Perumal, T. (2016). Improving anomalous rare attack detection rate for intrusion detection system using support vector machine and genetic programming. Neural Processing Letters, 44(2), 279-290.
1. Mohd Pozi, M. S., Sulaiman, M. N., Mustapha, N., & Perumal, T. (2015). A new classification model for a class imbalanced data set using genetic programming and support vector machines: case study for wilt disease classification. Remote Sensing Letters, 6(7), 568-577.
