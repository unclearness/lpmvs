#include "opencv2/highgui.hpp"

#include "orientation_2d.h"

// hairrecon::GaborParams params;

int ksize = 5;

int theta_pos = 0;
int theta_pos_max = 50;
double theta = 0;

int sigma_pos = 0;
int sigma_pos_max = 50;
double sigma = 2.0;

int lambd_pos = 0;
int lambd_pos_max = 50;
double lambd = 4.0;

int gamma_pos = 0;
int gamma_pos_max = 50;
double gamma = 0.75;

int psi_pos = 0;
int psi_pos_max = 50;
double psi = 0;

cv::Mat kernel;
cv::Mat src;
cv::Mat src_inv;
cv::Mat dst, dst_inv;
cv::Mat vis_dst, vis_dst_inv;

void cbCommon() {
  kernel = cv::getGaborKernel(cv::Size(ksize, ksize), sigma, theta, lambd,
                              gamma, psi);

  cv::filter2D(src, dst, CV_8UC1, kernel);
  cv::filter2D(src_inv, dst_inv, CV_8UC1, kernel);

  cv::normalize(dst, vis_dst, 255, 0, cv::NormTypes::NORM_MINMAX, CV_8UC1);
  cv::normalize(dst, vis_dst_inv, 255, 0, cv::NormTypes::NORM_MINMAX, CV_8UC1);

  cv::resize(kernel, kernel, cv::Size(256, 256), 0, 0,
             cv::InterpolationFlags::INTER_NEAREST);
}

void cbKsize(int pos, void* userdata) {
  ksize = pos;
  cbCommon();
}

void cbSigma(int pos, void* userdata) {
  sigma = pos / 2;
  cbCommon();
}

void cbLambd(int pos, void* userdata) {
  lambd = pos / 2;
  cbCommon();
}

void cbTheta(int pos, void* userdata) {
  theta = CV_PI *  pos / theta_pos_max;
  cbCommon();
}

void cbGamma(int pos, void* userdata) {
  gamma = pos / 10;
  cbCommon();
}

void cbPsi(int pos, void* userdata) {
  psi = 2 * CV_PI * pos / psi_pos_max;
  cbCommon();
}

int main() {

  src = cv::imread("../data/ayasa.jpg", cv::ImreadModes::IMREAD_GRAYSCALE);
  src_inv = 255 - src;  

  cbCommon();

  cv::namedWindow("gabor", cv::WINDOW_AUTOSIZE);
  cv::createTrackbar("size", "gabor", &ksize, 55, cbKsize);
  cv::createTrackbar("theta", "gabor", &theta_pos, theta_pos_max, cbTheta);
  cv::createTrackbar("lambd", "gabor", &lambd_pos, lambd_pos_max, cbLambd);
  cv::createTrackbar("sigma", "gabor", &sigma_pos, sigma_pos_max, cbSigma);
  cv::createTrackbar("gamma", "gabor", &gamma_pos, gamma_pos_max, cbGamma);
  cv::createTrackbar("psi", "gabor", &psi_pos, psi_pos_max, cbPsi);

  while (true) {
    cv::imshow("gabor", kernel);
    cv::imshow("src", src);

    cv::imshow("dst", vis_dst);
    cv::imshow("src_inv", src_inv);
    cv::imshow("dst_inv", vis_dst_inv);
    cv::waitKey(1);
  }

  return 0;
}