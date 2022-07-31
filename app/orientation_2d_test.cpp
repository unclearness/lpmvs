#include "orientation_2d.h"

#include "opencv2/highgui.hpp"

int main() {
  std::string data_dir = "../data/orientation_2d/";
  std::string data_path = data_dir + "lenna.jpg";
  cv::Mat1b gray = cv::imread(data_path, cv::ImreadModes::IMREAD_GRAYSCALE);

  hairrecon::EstimateHairOrientation2DParams params;
  // params.confidence_percentile_th = 0.4f;
  // params.gabor_params = hairrecon::GaborParams(12);

  params.debug = false;

  params.confidence_percentile_th = 0.95f;
  params.angle_split = 32;

  hairrecon::GaborParams gabor_params(21);
  params.gabor_params = gabor_params;
  // params.gabor_params.lambd = 5;
  params.confidence_percentile_th = 0.65f;

  cv::Mat1b mask = cv::Mat1b::ones(gray.size()) * 255;
  cv::Mat raw_input = cv::imread(data_path, -1);
  if (raw_input.channels() == 4) {
    std::vector<cv::Mat1b> splitted;
    cv::split(raw_input, splitted);
    mask = splitted[3];
  }
  cv::imwrite(data_dir + "/mask.png", mask);
  params.fg_mask = mask;

  gray = 255 - gray;

  cv::imwrite("input.jpg", gray);

  int mark_len = gray.rows / 10;
  auto mark = hairrecon::OrientatnToBgrMark(mark_len);
  cv::imwrite("mark.png", mark);

  hairrecon::EstimateHairOrientation2DOutput output;

  cv::Mat1f gray_d;
  gray.convertTo(gray_d, CV_32FC1);

  hairrecon::EstimateHairOrientation2D(gray_d, params, output);

  cv::Mat3b vis_orien, vis_orien_raw;
  cv::Mat1b vis_orien_gray;
  hairrecon::OrientationToBgr(output.orien, vis_orien);
  hairrecon::OrientationToBgr(output.orien_raw, vis_orien_raw);
  hairrecon::OrientationToGray(output.orien, vis_orien_gray,
                               params.angle_split);

  cv::Rect roi(0, gray.rows - mark_len - 1, mark_len, mark_len);
  mark.copyTo(vis_orien_raw(roi));
  mark.copyTo(vis_orien(roi));

  cv::imwrite("orien.png", vis_orien);
  cv::imwrite("orien_gray.png", vis_orien_gray);
  cv::imwrite("orien_raw.png", vis_orien_raw);
  cv::imwrite("high_confidence_mask.png", output.high_confidence_mask);
  cv::imwrite("fg_mask.png", output.fg_mask);

  // Iterative refinement: Input confidence
  // This process significantly improves 1st result
  hairrecon::EstimateHairOrientation2D(output.confidence.clone(), params,
                                       output);
  hairrecon::OrientationToBgr(output.orien, vis_orien);
  hairrecon::OrientationToBgr(output.orien_raw, vis_orien_raw);
  mark.copyTo(vis_orien_raw(roi));
  mark.copyTo(vis_orien(roi));
  cv::imwrite("orien2.png", vis_orien);
  cv::imwrite("orien_gray2.png", vis_orien_gray);
  cv::imwrite("orien_raw2.png", vis_orien_raw);
  cv::imwrite("high_confidence_mask2.png", output.high_confidence_mask);
  cv::imwrite("fg_mask2.png", output.fg_mask);

  hairrecon::EstimateHairOrientation2D(output.confidence.clone(), params,
                                       output);
  hairrecon::OrientationToBgr(output.orien, vis_orien);
  hairrecon::OrientationToBgr(output.orien_raw, vis_orien_raw);
  mark.copyTo(vis_orien_raw(roi));
  mark.copyTo(vis_orien(roi));
  cv::imwrite("orien3.png", vis_orien);
  cv::imwrite("orien3_gray.png", vis_orien_gray);
  cv::imwrite("orien_raw3.png", vis_orien_raw);
  cv::imwrite("high_confidence_mask3.png", output.high_confidence_mask);
  cv::imwrite("fg_mask3.png", output.fg_mask);
  return 0;
}
