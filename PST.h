#ifndef PST_H
#define PST_H

#include<opencv2/opencv.hpp>
#include<iostream>
#include<vector>

using namespace std;
using namespace cv;

struct Handles {
	double LPF;
	double Phase_strength;
	double Warp_strength;
	double Thresh_min;
	double Thresh_max;
};

void PST(Mat I, Handles handles, bool Morph_flag, Mat & out, Mat & PST_Kernel);

vector<double> linspace(double a, double b, int n);

void tdgrid(std::vector<double> r, std::vector<double> c, Mat & rMat, Mat & cMat);

void cart2pol(Mat x, Mat y, Mat & theta, Mat & rho);

void fftshift(Mat & in, Mat & out);

void exp(Mat & in, Mat & out, std::complex<double> mult);

Mat pow(Mat in, int p);

Mat atan(Mat in);

Mat log(Mat in);

Mat phase(Mat re, Mat im);

Mat complexMul(Mat A, Mat B);

#endif