% WPD : image wrapped phase denoising.

# ABOUT

* Author    : Ivan de Jesus May-Cen <ivan.mc@progreso.tecnm.mx>
* Copyright : 
* Licence   : 

# OVERVIEW

This source code provides an implementation of the image wrapped phase denoising.

# UNIX/LINUX USER GUIDE

The code is compilable on Unix/Linux. 

- Compilation. 
g++ -O2 wpdPFTVtest.cpp -o test -lrt -lblitz `pkg-config --cflags opencv4` `pkg-config --libs opencv4`

or

g++ -O2 wpdPFTVtest.cpp -o test -lrt -lblitz `pkg-config --cflags opencv` `pkg-config --libs opencv`

if you use minor version of the openCV4.

- Library. 
This code requires the openCV library and the blitz++ library.

- Image format. 
Only the PGM format is supported for the output. The resulting images are saved in the folder "imagenes".

-------------------------------------------------------------------------
Usage:
1. Download the code package and extract it. Go to that directory. 

2. Compile the source code (on Unix/Linux). 

3. Run WPD image denoising.
./test
The generic way to run the code is:

./test opcion

where "option" is a integer:
0 : experiment with sinthetic data
1 : experiment with real data, "brain.jpeg" is the processed image

-------------------------------------------------------------------------
     PARALLEL/SERIAL VERSION
         June, 7Th, 2023

The serial and parallel implementation codes for double and float precision are attached.
To download from:
          parallel.zip
          serial.zip

The processed images can be downloaded from:

https://tinyurl.com/3zf3v6ht

# ABOUT THIS FILE

Copyright 2023

*"brain.jpeg" was an image obtained from Ghiglia and Pritt (1998).

Ghiglia, D. C., & Pritt, M. D. (1998). Two-dimensional phase unwrapping: theory, algorithms, and software. A Wiley Interscience Publication.
