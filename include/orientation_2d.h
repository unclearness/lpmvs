#pragma once
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#include <string>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

namespace hairrecon {

struct GaborParams {
  int ksize = 12;  // kernel size. emprically set 3 times larger than lambd

  /* Same paramters to "Single-View Hair Modeling for Portrait Manipulation"  */
  double lambd = 4;  // IMPORTANT: wavelength of the sinusoidal factor.
  double sigma =
      1.8;  // Standard deviation of the gaussian envelope. Decide span
            // of the sin wave. Meaningful maximum value is ksize.
  double gamma = 0.75;  // spacial aspect ratio. Typically less than 1.0
  // black line on white background -> PI
  // white line on black background -> 0.0
  double psi = 0.0;
  /* Same paramters to "Single-View Hair Modeling for Portrait Manipulation"  */

  cv::Mat GetKernel(double theta) const {
    return cv::getGaborKernel(cv::Size(ksize, ksize), sigma, theta, lambd,
                              gamma, psi);
  }

  GaborParams() = default;
  
  // Heuristically set parameters
  GaborParams(int ksize)
      : ksize(ksize),
        lambd(ksize / 3.0),
        sigma(lambd / 4.0 * 1.8),
        gamma(0.75), psi(0.0) {}

};

enum class ConfidenceType {
  RESPONSE_VARIANCE,
  LINE_VARIANCE
};


struct EstimateHairOrientation2DParams {
  int angle_split = 32;
  // IMPORTANT: This depends on how much hairs are masked in advance
  float confidence_percentile_th = 0.25f;
  cv::Mat1b fg_mask;  // if empty, calculated automatically
  GaborParams gabor_params;
  bool debug = false;
  std::string debug_dir = "./";

  ConfidenceType confidence_type = ConfidenceType::RESPONSE_VARIANCE;

};

struct AngleResponse {
  float radian;
  cv::Mat1f response;
};

struct EstimateHairOrientation2DOutput {
  std::vector<AngleResponse> responses;
  cv::Mat1b fg_mask;
  cv::Mat1f max_response;
  cv::Mat1i orien_id_raw;
  cv::Mat1f orien_raw;
  cv::Mat1f orien;
  cv::Mat1f confidence;
  float confidence_th;
  cv::Mat1b high_confidence_mask;

  void Clear() {
    responses.clear();
    fg_mask = cv::Mat1b();
    max_response = cv::Mat1f();
    orien_id_raw = cv::Mat1i();
    orien_raw = cv::Mat1f();
    orien = cv::Mat1f();
    confidence = cv::Mat1f();
    confidence_th = 0.0f;
    high_confidence_mask = cv::Mat1b();
  }
};

// gray: background -> 0, foreground -> 1-255
bool EstimateHairOrientation2D(const cv::Mat1f& gray,
                               const EstimateHairOrientation2DParams& params,
                               EstimateHairOrientation2DOutput& output);

bool OrientationToHsv(const cv::Mat1f& orien, cv::Mat3b& vis_hsv,
                    bool is_degree = false, bool semicircle = true,
                    unsigned char s = 255, unsigned char v = 255);
bool OrientationToBgr(const cv::Mat1f& orien, cv::Mat3b& vis_orien,
                    bool is_degree = false, bool semicircle = true,
                    unsigned char s = 255, unsigned char v = 255);
bool OrientationToGray(const cv::Mat1f& orien, cv::Mat1b& vis_orien,
                       int angle_split);

cv::Vec3b AngleToHsv(float rad, bool is_degree = false, bool semicircle = true,
                     unsigned char s = 255, unsigned char v = 255);

cv::Vec3b AngleToBgr(float rad, bool is_degree = false, bool semicircle = true,
                     unsigned char s = 255, unsigned char v = 255);

cv::Mat3b OrientatnToBgrMark(int r, bool from_y_axis = true,
                             bool clockwise = true, bool semicircle = true,
                             unsigned char s = 255, unsigned char v = 255);

unsigned char AngleToGray(float rad, int angle_split);

}  // namespace hairrecon