#include<opencv2/opencv.hpp>
#include<math.h>
#include "PST.h"

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
	exp(-pow((RHO / sqrt(sigma)), 2), temp);
	fftshift(temp, temp);
	planes[0] = planes[0].mul(temp);
	planes[1] = planes[1].mul(temp);
	merge(planes, 2, Image_orig_f);
	Mat Image_orig_filtered;
	idft(Image_orig_f, Image_orig_filtered, cv::DFT_SCALE | cv::DFT_INVERSE);
	split(Image_orig_filtered, planes);
	planes[1] = Mat::zeros(planes[1].size(), CV_64FC1);
	merge(planes, 2, Image_orig_filtered);

	// Construct the PST Kernel
	PST_Kernel = (RHO * handles.Warp_strength).mul(atan(RHO * handles.Warp_strength)) - 0.5 * log(1 + pow(RHO * handles.Warp_strength, 2));
	double min, max;
	minMaxLoc(PST_Kernel, &min, &max);
	PST_Kernel = PST_Kernel / max * handles.Phase_strength;
	
	// Apply the PST Kernel (problem)
	dft(Image_orig_filtered, temp);
	Mat jkern, shift; // [] = { Mat::zeros(PST_Kernel.size(), CV_64FC1), Mat::zeros(PST_Kernel.size(), CV_64FC1) };
	exp(PST_Kernel, jkern, -k);
	fftshift(jkern, shift);
	temp = complexMul(temp, shift);
	Mat Image_orig_filtered_PST;
	idft(temp, Image_orig_filtered_PST, cv::DFT_SCALE | cv::DFT_INVERSE);

	// Calculate phase of the transformed image
	split(Image_orig_filtered_PST, planes);
	Mat PHI_features = phase(planes[0], planes[1]);

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

void tdgrid(std::vector<double> r, std::vector<double> c, Mat & rMat, Mat & cMat) {
	int rows = r.size();
	int cols = c.size();
	rMat = Mat(rows, cols, CV_64FC1);
	cMat = Mat(rows, cols, CV_64FC1);
	for (int i = 0; i < rows; i++) {
		double* ri = rMat.ptr<double>(i);
		double* ci = cMat.ptr<double>(i);
		for (int j = 0; j < cols; j++) {
			ri[j] = r[i];
			ci[j] = c[j];
		}
	}
}

void cart2pol(Mat x, Mat y, Mat & theta, Mat & rho) {
	int rows = x.rows;
	int cols = x.cols;
	theta = Mat(rows, cols, CV_64FC1);
	rho = Mat(rows, cols, CV_64FC1);
	for (int i = 0; i < rows; i++) {
		double* ti = theta.ptr<double>(i);
		double* ri = rho.ptr<double>(i);
		for (int j = 0; j < cols; j++) {
			ti[j] = atan2(y.at<double>(i, j), x.at<double>(i, j));
			ri[j] = hypot(x.at<double>(i, j), y.at<double>(i, j));
		}
	}
}

void fftshift(Mat & in, Mat & out) {
	out = in.clone();
	int mx1, my1, mx2, my2;
	mx1 = out.cols / 2;
	my1 = out.rows / 2;
	mx2 = int(ceil(out.cols / 2.0));
	my2 = int(ceil(out.rows / 2.0));
	Mat q0(out, Rect(0, 0, mx2, my2));
	Mat q1(out, Rect(mx2, 0, mx1, my2));
	Mat q2(out, Rect(0, my2, mx2, my1));
	Mat q3(out, Rect(mx2, my2, mx1, my1));
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q2.copyTo(tmp);
	q1.copyTo(q2);
	tmp.copyTo(q1);
	vconcat(q1, q3, out);
	vconcat(q0, q2, tmp);
	hconcat(tmp, out, out);
}

void exp(Mat & in, Mat & out, std::complex<double> mult) {
	int rows = in.rows;
	int cols = in.cols;
	Mat re = Mat(rows, cols, CV_64FC1);
	Mat im = Mat(rows, cols, CV_64FC1);
	for (int i = 0; i < rows; i++) {
		double * rei = re.ptr<double>(i);
		double * imi = im.ptr<double>(i);
		double * ini = in.ptr<double>(i);
		for (int j = 0; j < cols; j++) {
			std::complex<double> val = exp(ini[j] * mult);
			rei[j] = real(val);
			imi[j] = imag(val);
		}
	}
	Mat temp[] = { re, im };
	merge(temp, 2, out);
}

Mat pow(Mat in, int p) {
	int rows = in.rows;
	int cols = in.cols;
	for (int i = 0; i < rows; i++) {
		double * ini = in.ptr<double>(i);
		for (int j = 0; j < cols; j++) {
			ini[j] = pow(ini[j], p);
		}
	}
	return in;
}

Mat atan(Mat in) {
	int rows = in.rows;
	int cols = in.cols;
	for (int i = 0; i < rows; i++) {
		double * ini = in.ptr<double>(i);
		for (int j = 0; j < cols; j++) {
			ini[j] = atan(ini[j]);
		}
	}
	return in;
}


Mat log(Mat in) {
	int rows = in.rows;
	int cols = in.cols;
	for (int i = 0; i < rows; i++) {
		double * ini = in.ptr<double>(i);
		for (int j = 0; j < cols; j++) {
			ini[j] = log(ini[j]);
		}
	}
	return in;
}

Mat phase(Mat re, Mat im) {
	int rows = re.rows;
	int cols = re.cols;
	for (int i = 0; i < rows; i++) {
		double * rei = re.ptr<double>(i);
		double * imi = im.ptr<double>(i);
		for (int j = 0; j < cols; j++) {
			rei[j] = atan2(imi[j], rei[j]);
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