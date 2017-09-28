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

void tdgrid(vector<double> x, vector<double> y, Mat & xOut, Mat & yOut);

void cart2pol(Mat x, Mat y, Mat & theta, Mat & rho);

Mat fftshift(Mat in);

Mat matexp(Mat in);

void splitmatexp(Mat input, Mat re, Mat im, complex<double> mult);

Mat matpow(Mat in, int p);

Mat matatan(Mat in);

Mat matlog(Mat in);

Mat matphase(Mat re, Mat im);

Mat complexMul(Mat A, Mat B);

#endif