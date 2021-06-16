This is C++ based implementation for SVGPM. 

In order to build this program, please install shark machine learning library version 3 into your system.

This program has been sucessfully build and run using WSL2 Ubuntu Based Distro.

In order to run the program (once it has been built):
    go to /build/app
    run svgpm on command line/console:
        svgpm  -gen    100	-p	100	-ds  wine_quality.csv.

The program will run for 100 generation (-gen 100), with population 100 (-p 100), on dataset wine_quality.csv (-ds wine_quality.csv) using 5-fold cross-validation.