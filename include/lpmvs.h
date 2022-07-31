#pragma once

#include "ugu/camera.h"
#include "ugu/image.h"
#include "ugu/line.h"

namespace hairrecon {

struct Keyframe;
using KeyframePtr = std::shared_ptr<Keyframe>;
using KeyframeConstPtr = std::shared_ptr<const Keyframe>;

constexpr double MAX_COST = std::numeric_limits<double>::max() * 0.0001;

struct Keyframe {
  int id;

  // std::string color_path;
  // ugu::Image3b color;
  // std::string mask_path;
  // ugu::Image1b mask;

  // From 2d orientation estimation
  ugu::Image1f intensity;
  ugu::Image1f orientation2d;
  ugu::Image1f confidence;

  // neighboring views
  std::vector<KeyframePtr> neighbor_kfs;

  ugu::Image1f cost;
  // [0]: depth value, [1][2]: 3D direction in polar coordiantes (theta and phi)
  ugu::Image3f line3d;
  // Calculate for each neighbor_kfs
  // std::vector<ugu::Image3f> line3ds;
  // std::vector<ugu::Image1f> depths;

  ugu::Image1b filtered_mask;
  ugu::Image1f filtered_cost;
  ugu::Image3f filtered_line3d;

  // Camera parameter of this view
  std::shared_ptr<const ugu::LinePinholeCamera> camera;

  static KeyframePtr Create() { return std::make_shared<Keyframe>(); }

  Keyframe(){};
  ~Keyframe(){};
};

struct AngleConfig {
  bool is_radian = true;
  bool from_y_axis = true;
  bool clockwise = true;
};

enum class RandomInitType { kUniform, kAlignOrien };

struct LpmvsParams {
  float alpha = 0.1f;
  int iter = 8;
  bool skip_low_confidence = true;
  bool alternately_reverse = true;
  bool spatial_propagation = true;
  bool random_search = true;
  bool colmap_search = true;
  float perturbation_depth = 0.01f;
  float perturbation_direc_theta_rad = ugu::radians(3.0f);
  float perturbation_direc_phi_rad = ugu::radians(3.0f);
  float perturbation_direc_axis_rad = ugu::radians(3.0f);

  int sample_2d_k = 41;
  float sample_2d_r = 10.0f;

  float min_depth = 1.5f;
  float max_depth = 3.0f;
  int random_seed = 0;

  std::string debug_dir = "./";
  int debug_row_interval = 100;

  bool red_black_propagation = true;
  bool red_black_fast = true;

  RandomInitType random_init_type = RandomInitType::kAlignOrien;
  RandomInitType random_search_type = RandomInitType::kUniform;
};

bool LpmvsInitializeRandomEngine(const LpmvsParams& params);

bool LineBasedPatchMatchMVS(KeyframePtr kf, const LpmvsParams& params);
bool LineBasedPatchMatchMVS(std::vector<KeyframePtr>& kfs,
                            const LpmvsParams& params);

bool LineFiltering(KeyframePtr kf, float pos_th = 0.001f,
                   float angle_th = ugu::radians(10.0f), int match_view_th = 2);
bool LineFiltering(std::vector<KeyframePtr>& kfs, float pos_th = 0.001f,
                   float angle_th = ugu::radians(10.0f), int match_view_th = 2);

bool ColorizeLine3d(std::shared_ptr<const ugu::LinePinholeCamera>& camera,
                    const cv::Mat3f& line3d, cv::Mat3b& vis_line3d);
bool Cost2Gray(const cv::Mat1f& cost, cv::Mat1b& vis_cost,
               float min_cost = -1.f, float max_cost = -1.f);

struct CostStats {
  std::vector<float> data;
  float mean, median, stdev, max, min;
  float valid_ratio;

  std::string ToString() const {
    std::ostringstream sout;
    sout << "# Valid Data  " << data.size() << std::endl;
    sout << "# Valid Ratio " << valid_ratio << std::endl;
    sout << "Mean          " << mean << std::endl;
    sout << "Median        " << median << std::endl;
    sout << "Stdev         " << stdev << std::endl;
    sout << "Max           " << max << std::endl;
    sout << "Min           " << min << std::endl;
    return sout.str();
  }
};

CostStats GetCostStats(const cv::Mat1f& cost, bool ignore_invalid = true);

bool DebugDump(const std::string& dir, const std::string& prefix,
               KeyframePtr kf);

ugu::Line3d GetLine3dInCameraCoord(
    std::shared_ptr<const ugu::LinePinholeCamera> camera, double x, double y,
    double depth, double theta, double phi);

void GetLineParam(const ugu::Line3d& camera_line3d, double& depth,
                  double& theta, double& phi);

}  // namespace hairrecon