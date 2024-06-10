# Volume render of a 3D datacube

<div style="display: flex; justify-content: space-around;">
  <img src="Serial/images/volumerender0.png" alt="Volume Render 0" width="30%" />
  <img src="Serial/images/volumerender0.png" alt="Volume Render 1" width="30%" />
</div>

## Overview

## Introduction

## Dependencies
This project was implemented with the following dependencies, which are the latest:
- Eigen: Version 3.4.0
- OpenCV: Version 4.9.0
- HDF5: Version 1.14.3
- GoogleTest: Version 1.14.0

To install the dependenices run on the project root directory:
'''
chmod +x ./dependencies.sh
'''

## Compiling and running the serial code
To compile the serial program and the tests:
'''
$ make 
'''

To run the serial program:
'''
$ make run ARG
'''
Where arg is the amount of pictures wished to process. The default is 10 images.

To run the tests:
'''
$ make run tests
'''

To remove the program and test compilation:
'''
$ make clean
'''




