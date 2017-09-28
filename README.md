# Phase-Stretch-Transform-CPP

This is a C++ implementation of the Phase Stretch Transform algorithm developed at UCLA. This algorithm finds features in an image, and can create a binary output image illustrating sharp contrasts within the original image. The [original MatLab implementation](https://github.com/JalaliLabUCLA/Image-feature-detection-using-Phase-Stretch-Transform) has a much more in-depth explanation of how the algorithm works, and more information is also available on [Wikipedia](https://en.wikipedia.org/wiki/Phase_stretch_transform). The test.cpp file contains an example use case, the output of which is shown below:

![Example](https://github.com/haydengunraj/Phase-Shift-Transform-CPP/blob/master/output.png "Example")

## Dependencies

OpenCV is required to use this implementation, and it was developed with OpenCV 3.3.0.

## Copyright

Although I developed this implementation, the algorithm was developed and first implemented in MatLab at the Jalali Lab at the University of California, Los Angeles (UCLA). PST is a spin-off from research on the photonic time stretch technique done at this lab. More information about the technique can be found on the group's [website](http://www.photonics.ucla.edu).

This function is provided for research purposes only. A license must be obtained from the University of California, Los Angeles for any commercial  applications. The software is protected under a US patent.

## Citations

1. M. H. Asghari, and B. Jalali, "Edge detection in digital images using dispersive phase stretch," International Journal of Biomedical Imaging, Vol. 2015, Article ID 687819, pp. 1-6 (2015).
2. M. H. Asghari, and B. Jalali, "Physics-inspired image edge detection," IEEE Global Signal and Information Processing Symposium (GlobalSIP 2014), paper: WdBD-L.1, Atlanta, December 2014.