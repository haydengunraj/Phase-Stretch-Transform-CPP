#include<opencv2/opencv.hpp>
#include<math.h>
#include "PST.h"

using namespace std;
using namespace cv;

int main() {
	// Load image in grayscale
	Mat I = imread("lena_gray_512.tif", CV_LOAD_IMAGE_GRAYSCALE);
	
	// Set up handles
	Handles handles;
	handles.LPF = 0.5;
	handles.Phase_strength = 0.48;
	handles.Warp_strength = 12.14;
	handles.Thresh_min = -1;
	handles.Thresh_max = 0.004;

	// Convert to float
	I.convertTo(I, CV_64FC1);

	// Call PST on image, store PST and PST_Kernel
	Mat pst, PST_Kernel;
	PST(I, handles, true, pst, PST_Kernel);

	// Show result
	imshow("Output", pst);
	waitKey(0);

	// Show the PST phase kernel gradient
	Mat D_PST_Kernel_x, D_PST_Kernel_y;
	// Yet to be implemented

	system("pause");
	return 0;
}