#include<opencv2/opencv.hpp>
#include<math.h>
#include "PST.h"

using namespace std;
using namespace cv;

const complex<double> k(0, 1);

void PST(Mat I, Handles handles, bool Morph_flag, Mat & out, Mat & PST_Kernel) {
	// Define two dimentional cartesian (rectangular) Mats, X and Y
	double L = 0.5;
	vector<double> x = linspace(-L, L, I.rows); // UCLA did rows
	vector<double> y = linspace(-L, L, I.cols); // UCLA did cols
	Mat X(x.size(), y.size(), CV_64FC1);
	Mat Y(x.size(), y.size(), CV_64FC1);
	tdgrid(x, y, X, Y);

	// Convert cartesian X and Y vectors to polar vectors, THETA and RHO
	Mat THETA(X.size(), CV_64FC1);
	Mat RHO(Y.size(), CV_64FC1);
	cart2pol(X, Y, THETA, RHO);
	// Define two dimensional cartesian frequency vectors, FX and FY
	double X_step = x[1] - x[0];
	vector<double> fx = linspace(-L / X_step, L / X_step, x.size());
	double fx_step = fx[1] - fx[0];
	double Y_step = y[1] - y[0];
	vector<double> fy = linspace(-L / Y_step, L / Y_step, y.size());
	double fy_step = fy[1] - fy[0];
	Mat FX(fx.size(), fy.size(), CV_64FC1);
	Mat FY(fx.size(), fy.size(), CV_64FC1);
	tdgrid(fx, fy, FX, FY);

	// Convert cartesian vectors (FX and FY) to polar vectors, FTHETA and FRHO
	Mat FTHETA(FX.size(), CV_64FC1);
	Mat FRHO(FY.size(), CV_64FC1);
	cart2pol(FX, FY, FTHETA, FRHO);
	
	// Low pass filter the original image to reduce noise
	Mat planes[] = { Mat_<double>(I), Mat::zeros(I.size(), CV_64FC1) };
	Mat Image_orig_f, temp;
	merge(planes, 2, Image_orig_f);
	dft(Image_orig_f, Image_orig_f);
	split(Image_orig_f, planes);
	double sigma = (handles.LPF * handles.LPF) / log(2);
	planes[0] = planes[0].mul(fftshift(matexp(-matpow((RHO / sqrt(sigma)), 2))));
	planes[1] = planes[1].mul(fftshift(matexp(-matpow((RHO / sqrt(sigma)), 2))));
	merge(planes, 2, Image_orig_f);
	Mat Image_orig_filtered;
	idft(Image_orig_f, Image_orig_filtered, cv::DFT_SCALE | cv::DFT_INVERSE);
	split(Image_orig_filtered, planes);
	planes[1] = Mat::zeros(planes[1].size(), CV_64FC1);
	merge(planes, 2, Image_orig_filtered);

	// Construct the PST Kernel
	PST_Kernel = (RHO * handles.Warp_strength).mul(matatan(RHO * handles.Warp_strength)) - 0.5 * matlog(1 + matpow(RHO * handles.Warp_strength, 2));
	double min, max;
	minMaxLoc(PST_Kernel, &min, &max);
	PST_Kernel = PST_Kernel / max * handles.Phase_strength;
	
	// Apply the PST Kernel (problem)
	dft(Image_orig_filtered, temp);
	Mat jkern[] = { Mat::zeros(PST_Kernel.size(), CV_64FC1), Mat::zeros(PST_Kernel.size(), CV_64FC1) };
	splitmatexp(PST_Kernel, jkern[0], jkern[1], -k);
	Mat shift;
	merge(jkern, 2, shift);
	temp = complexMul(temp, fftshift(shift));
	Mat Image_orig_filtered_PST;
	idft(temp, Image_orig_filtered_PST, cv::DFT_SCALE | cv::DFT_INVERSE);

	// Calculate phase of the transformed image
	split(Image_orig_filtered_PST, planes);
	Mat PHI_features = matphase(planes[0], planes[1]);

	if (!Morph_flag) {
		out = PHI_features;
	}
	else {
		// Find image sharp transitions by thresholding the phase
		Mat features = Mat::zeros(PHI_features.size(), CV_64FC1);
		int xSize = features.cols;
		int ySize = features.rows;
		minMaxLoc(I, &min, &max);
		for (int i = 0; i < ySize; i++) {
			for (int j = 0; j < xSize; j++) {
				if (PHI_features.at<double>(i, j) > handles.Thresh_max || PHI_features.at<double>(i, j) < handles.Thresh_min) {
					features.at<double>(i, j) = 1;
				}
				if (I.at<double>(i, j) < max / 20) {
					features.at<double>(i, j) = 0;
				}
			}
		}

		// MatLab code does cleaning here, but this is not easy to port

		out = features;
	}
}

vector<double> linspace(double a, double b, int n) {
	vector<double> arr;
	double st = (b - a) / (n - 1);
	while (a <= b) {
		arr.push_back(a);
		a += st;
	}
	return arr;
}

void tdgrid(vector<double> x, vector<double> y, Mat & xOut, Mat & yOut) {
	int xSize = x.size();
	int ySize = y.size();
	for (int i = 0; i < xSize; i++) {
		for (int j = 0; j < ySize; j++) {
			xOut.at<double>(i, j) = x[i];
			yOut.at<double>(i, j) = y[j];
		}
	}
}

void cart2pol(Mat x, Mat y, Mat & theta, Mat & rho) {
	int xSize = x.cols;
	int ySize = x.rows;
	for (int i = 0; i < ySize; i++) {
		for (int j = 0; j < xSize; j++) {
			theta.at<double>(i, j) = atan2(y.at<double>(i, j), x.at<double>(i, j));
			rho.at<double>(i, j) = hypot(x.at<double>(i, j), y.at<double>(i, j));
		}
	}
}

Mat fftshift(Mat in) {
	int mxl = in.cols / 2;
	int myl = in.rows / 2;
	int mxh = ceil(in.cols / 2.0);
	int myh = ceil(in.rows / 2.0);
	Mat q0(in, Rect(0, 0, mxh, myh));
	Mat q1(in, Rect(mxh, 0, mxl, myh));
	Mat q2(in, Rect(0, myh, mxh, myl));
	Mat q3(in, Rect(mxh, myh, mxl, myl));
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q2.copyTo(tmp);
	q1.copyTo(q2);
	tmp.copyTo(q1);
	vconcat(q1, q3, in);
	vconcat(q0, q2, tmp);
	hconcat(tmp, in, in);
	return in;
}

Mat matexp(Mat in) {
	int xSize = in.cols;
	int ySize = in.rows;
	for (int i = 0; i < ySize; i++) {
		for (int j = 0; j < xSize; j++) {
			in.at<double>(i, j) = exp(in.at<double>(i, j));
		}
	}
	return in;
}

void splitmatexp(Mat input, Mat re, Mat im, complex<double> mult) {
	int xSize = input.cols;
	int ySize = input.rows;
	for (int i = 0; i < ySize; i++) {
		for (int j = 0; j < xSize; j++) {
			complex<double> val = exp(input.at<double>(i, j) * mult);
			re.at<double>(i, j) = real(val);
			im.at<double>(i, j) = imag(val);
		}
	}
}

Mat matpow(Mat in, int p) {
	int xSize = in.cols;
	int ySize = in.rows;
	for (int i = 0; i < ySize; i++) {
		for (int j = 0; j < xSize; j++) {
			in.at<double>(i, j) = pow(in.at<double>(i, j), p);
		}
	}
	return in;
}

Mat matatan(Mat in) {
	int xSize = in.cols;
	int ySize = in.rows;
	for (int i = 0; i < ySize; i++) {
		for (int j = 0; j < xSize; j++) {
			in.at<double>(i, j) = atan(in.at<double>(i, j));
		}
	}
	return in;
}

Mat matlog(Mat in) {
	int xSize = in.cols;
	int ySize = in.rows;
	for (int i = 0; i < ySize; i++) {
		for (int j = 0; j < xSize; j++) {
			in.at<double>(i, j) = log(in.at<double>(i, j));
		}
	}
	return in;
}

Mat matphase(Mat re, Mat im) {
	int xSize = re.cols;
	int ySize = re.rows;
	for (int i = 0; i < ySize; i++) {
		for (int j = 0; j < xSize; j++) {
			re.at<double>(i, j) = atan2(im.at<double>(i, j), re.at<double>(i, j));
		}
	}
	return re;
}

Mat complexMul(Mat A, Mat B) {
	Mat Acomp[] = { Mat::zeros(A.size(), CV_64FC1), Mat::zeros(A.size(), CV_64FC1) };
	Mat Bcomp[] = { Mat::zeros(B.size(), CV_64FC1),Mat::zeros(B.size(), CV_64FC1) };
	Mat Ccomp[] = { Mat::zeros(A.size(), CV_64FC1), Mat::zeros(A.size(), CV_64FC1) };
	split(A, Acomp);
	split(B, Bcomp);
	Ccomp[0] = Acomp[0].mul(Bcomp[0]) - Acomp[1].mul(Bcomp[1]);
	Ccomp[1] = Acomp[0].mul(Bcomp[1]) + Acomp[1].mul(Bcomp[0]);
	Mat C;
	merge(Ccomp, 2, C);
	return C;
}