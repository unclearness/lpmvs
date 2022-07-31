

#include "orientation_2d.h"
#include "util.h"

#include "opencv2/imgproc.hpp"

#include <cmath>

// for debug
#include <iostream>
#include "opencv2/imgcodecs.hpp"

namespace hairrecon {
bool EstimateHairOrientation2D(const cv::Mat1f& gray,
                               const EstimateHairOrientation2DParams& params,
                               EstimateHairOrientation2DOutput& output) {
  output.Clear();

  auto& responses = output.responses;
  responses.resize(params.angle_split);

   // cv::Mat1f laplacian;
   // cv::Laplacian(gray, laplacian, CV_32FC1);
   // cv::imwrite("lap.png", laplacian);
   // cv::imwrite("lap_mask.png", laplacian > 50);

  // Get responses for each angle
#pragma omp parallel for
  for (int i = 0; i < params.angle_split; i++) {
    AngleResponse response;

    float theta_deg =
        180.0f * static_cast<float>(i) / static_cast<float>(params.angle_split);

    float theta_rad = static_cast<float>(CV_PI / 180.0f * theta_deg);

    response.radian = theta_rad;

    auto kernel = params.gabor_params.GetKernel(theta_rad);

    cv::filter2D(gray, response.response, CV_32F, kernel);

    responses[i] = response;

    {
      // double resmax, resmin;
      // cv::minMaxLoc(response.response, &resmin, &resmax);
      // cv::normalize(response.response, response.response, 0, 1,
      // cv::NORM_MINMAX);
    }

    if (params.debug) {
      double resmax, resmin;
      cv::minMaxLoc(response.response, &resmin, &resmax);
      std::cout << resmin << " " << resmax << std::endl;
      cv::Mat1b vis_response;
      // normalize
      auto scale = 1.0 / (resmax - resmin);
      response.response.convertTo(vis_response, CV_8U, 255 * scale,
                                  -255 * resmin * scale);
      cv::imwrite(params.debug_dir + std::to_string(i) + "_" +
                      std::to_string(static_cast<int>(theta_deg)) + ".png",
                  vis_response);

      cv::Mat1b vis_kernel;
      cv::normalize(kernel, vis_kernel, 255.0, 0.0, cv::NormTypes::NORM_MINMAX,
                    CV_8UC1);
      cv::resize(vis_kernel, vis_kernel, cv::Size(128, 128), 0.0, 0.0,
                 cv::InterpolationFlags::INTER_NEAREST);
      cv::imwrite(params.debug_dir + std::to_string(i) + "_" +
                      std::to_string(static_cast<int>(theta_deg)) +
                      "_kernel.png",
                  vis_kernel);
    }
  }

  // Find the maximum response for each pixels
  auto& orien_raw = output.orien_raw;
  orien_raw = cv::Mat1f(gray.size());
  orien_raw.setTo(-1.0f);  // set minus

  auto& orien_id_raw = output.orien_id_raw;
  orien_id_raw = cv::Mat1i(gray.size());
  orien_id_raw.setTo(-1);  // set minus

  auto& max_response = output.max_response;
  max_response = cv::Mat1f(gray.size());
  max_response.setTo(0.0f);
  const float ignore_th = 0.0f;

  auto& fg_mask = output.fg_mask;
  if (params.fg_mask.empty()) {
    fg_mask = gray > 0 * 255;
  } else {
    fg_mask = params.fg_mask.clone();
  }

//#pragma omp parallel for
  for (int y = 0; y < max_response.rows; y++) {
    for (int x = 0; x < max_response.cols; x++) {
      // ignore background
      if (fg_mask.at<unsigned char>(y, x) == 0) {
        continue;
      }

      float& cur_max = max_response.at<float>(y, x);
      float& cur_angle = orien_raw.at<float>(y, x);
      int& cur_id = orien_id_raw.at<int>(y, x);
      for (int i = 0; i < params.angle_split; i++) {
        auto& response = responses[i];
        float tmp = response.response.at<float>(y, x);
        if (tmp > ignore_th && tmp > cur_max) {
          cur_max = tmp;
          cur_angle = response.radian;
          cur_id = i;
        }
      }
    }
  }

  //cv::bilateralFilter(orien_raw.clone(), orien_raw, 7, 0.3, 7);

  // confidence calculation
  auto& confidence = output.confidence;
  confidence = cv::Mat1f::zeros(gray.size());
  std::vector<float> confidence_values;
  confidence_values.reserve(confidence.total());

  //if (params.confidence_type == ConfidenceType::RESPONSE_VARIANCE) {
 // }
  for (int y = 0; y < confidence.rows; y++) {
    for (int x = 0; x < confidence.cols; x++) {
      // ignore background
      if (fg_mask.at<unsigned char>(y, x) == 0) {
        continue;
      }

      const float& max_res = max_response.at<float>(y, x);
      const float& max_angle = orien_raw.at<float>(y, x);
      const int& max_id = orien_id_raw.at<int>(y, x);
      auto& variance = confidence.at<float>(y, x);
      for (int i = 0; i < params.angle_split; i++) {
        if (max_id == i) {
          continue;
        }
        float cur_angle = responses[i].radian;
        float cur_res = responses[i].response.at<float>(y, x);
        double d = CalcMinAngleDist(max_angle, cur_angle);
        float res_diff = (max_res - cur_res) * (max_res - cur_res);
        variance += static_cast<float>(std::sqrt(d * res_diff * res_diff));
      }
      confidence_values.push_back(variance);
    }
  }


  std::sort(confidence_values.begin(), confidence_values.end(),
            std::greater<float>{});

  size_t percentile_index = static_cast<size_t>(
      params.confidence_percentile_th * confidence_values.size());
  output.confidence_th = confidence_values[percentile_index];

  auto& high_confidence_mask = output.high_confidence_mask;

  high_confidence_mask = confidence > output.confidence_th;

  output.orien = -1.0f * cv::Mat1f::ones(gray.size());
  orien_raw.copyTo(output.orien, high_confidence_mask);

  if (params.debug) {
    cv::Mat1b vis_conf;
    cv::normalize(confidence, vis_conf, 255.0, 0.0, cv::NormTypes::NORM_MINMAX,
                  CV_8UC1);
    cv::imwrite(params.debug_dir + "confidence.png", vis_conf);
  }

  return true;
}

bool OrientationToHsv(const cv::Mat1f& orien, cv::Mat3b& vis_hsv,
                      bool is_degree, bool semicircle, unsigned char s,
                      unsigned char v) {
  vis_hsv = cv::Mat3b::zeros(orien.size());

  for (int y = 0; y < vis_hsv.rows; y++) {
    for (int x = 0; x < vis_hsv.cols; x++) {
      float angle = orien.at<float>(y, x);
      if (angle < 0.0f) {
        continue;
      }

      vis_hsv.at<cv::Vec3b>(y, x) =
          AngleToHsv(angle, is_degree, semicircle, s, v);
    }
  }

  return true;
}

bool OrientationToBgr(const cv::Mat1f& orien, cv::Mat3b& vis_orien,
                      bool is_degree, bool semicircle, unsigned char s,
                      unsigned char v) {
  cv::Mat3b hsv;
  OrientationToHsv(orien, hsv, is_degree, semicircle, s, v);

  cv::cvtColor(hsv, vis_orien, cv::COLOR_HSV2BGR);

  return true;
}

bool OrientationToGray(const cv::Mat1f& orien, cv::Mat1b& vis_gray,
                       int angle_split) {
  vis_gray = cv::Mat1b::zeros(orien.size());

  for (int y = 0; y < vis_gray.rows; y++) {
    for (int x = 0; x < vis_gray.cols; x++) {
      float angle = orien.at<float>(y, x);
      if (angle < 0.0f) {
        continue;
      }

      vis_gray.at<unsigned char>(y, x) = AngleToGray(angle, angle_split);
    }
  }

  return true;
}

cv::Vec3b AngleToHsv(float angle, bool is_degree, bool semicircle,
                     unsigned char s, unsigned char v) {
  constexpr float inv_2pi = static_cast<float>(1.0f / (2 * M_PI));
  constexpr float inv_360 = static_cast<float>(1.0f / 360.0);
  float inv_scale = is_degree ? inv_360 : inv_2pi;
  if (semicircle) {
    inv_scale *= 2.0f;
  }
  unsigned char h = static_cast<unsigned char>(angle * inv_scale * 179);

  return cv::Vec3b(h, s, v);
}

cv::Vec3b AngleToBgr(float angle, bool is_degree, bool semicircle,
                     unsigned char s_, unsigned char v_) {
  auto hsv = AngleToHsv(angle, is_degree, semicircle, s_, v_);
  cv::Vec3b bgr;
  // cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

  float h = hsv[0] / 179.0f;
  float s = hsv[1] / 255.0f;
  float v = hsv[2] / 255.0f;

  float r = v;
  float g = v;
  float b = v;
  if (s > 0.0f) {
    h *= 6.0f;
    int i = (int)h;
    float f = h - (float)i;
    switch (i) {
      default:
      case 0:
        g *= 1 - s * (1 - f);
        b *= 1 - s;
        break;
      case 1:
        r *= 1 - s * f;
        b *= 1 - s;
        break;
      case 2:
        r *= 1 - s;
        b *= 1 - s * (1 - f);
        break;
      case 3:
        r *= 1 - s;
        g *= 1 - s * f;
        break;
      case 4:
        r *= 1 - s * (1 - f);
        g *= 1 - s;
        break;
      case 5:
        g *= 1 - s;
        b *= 1 - s * f;
        break;
    }
  }

  bgr[0] = static_cast<unsigned char>(b * 255);
  bgr[1] = static_cast<unsigned char>(g * 255);
  bgr[2] = static_cast<unsigned char>(r * 255);

  return bgr;
}

cv::Mat3b OrientatnToBgrMark(int diameter, bool from_y_axis, bool clockwise,
                             bool semicircle, unsigned char s,
                             unsigned char v) {
  cv::Mat3b mark(diameter, diameter, cv::Vec3b(0, 0, 0));

  int center_x = diameter / 2;
  int center_y = diameter / 2;

  for (int y = 0; y < diameter; y++) {
    for (int x = 0; x < diameter; x++) {
      float r = static_cast<float>(std::sqrt((x - center_x) * (x - center_x) +
                                             (y - center_y) * (y - center_y)));
      if (r > diameter / 2) {
        continue;
      }
      float angle = 0.0f;
      if (from_y_axis) {
        angle = static_cast<float>(std::atan2(x - center_x, y - center_y));
      } else {
        angle = static_cast<float>(std::atan2(y - center_y, x - center_x));
      }

      if (clockwise) {
        angle *= -1.0f;
      }

      angle += static_cast<float>(M_PI);

      if (angle > M_PI) {
        angle -= static_cast<float>(M_PI);
      }

      mark.at<cv::Vec3b>(y, x) = AngleToBgr(angle, false, semicircle, s, v);
      // mark.at<cv::Vec3b>(y, x) = AngleToHsv(angle);
    }
  }

  // cv::cvtColor(mark, mark, cv::COLOR_HSV2BGR);

  return mark;
}

unsigned char AngleToGray(float rad, int angle_split) {
  return static_cast<unsigned char>(std::round(angle_split * rad / PI));
}

}  // namespace hairrecon