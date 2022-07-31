#include "lpmvs.h"

#include <iostream>
#include <random>

#include "opencv2/imgproc.hpp"
#include "orientation_2d.h"
#include "ugu/timer.h"
#include "ugu/util/io_util.h"
#include "util.h"

namespace {

std::default_random_engine g_engine;
std::uniform_real_distribution<float> g_depth_dist(0.f, 10.0f);
std::uniform_real_distribution<float> g_theta_dist(
    0.f, static_cast<float>(hairrecon::PI));
// Limit to 0-PI (originally 0-2PI) to remove ambiguity in XY plane
std::uniform_real_distribution<float> g_phi_dist(
    0.f, static_cast<float>(hairrecon::PI));

std::uniform_real_distribution<float> g_uniform_for_direc_dist(-1.f, 1.f);

std::uniform_real_distribution<float> g_perturb_depth_dist(0.f, 10.0f);
std::uniform_real_distribution<float> g_perturb_theta_dist(
    0.f, static_cast<float>(hairrecon::PI));
std::uniform_real_distribution<float> g_perturb_phi_dist(
    0.f, static_cast<float>(hairrecon::PI));
std::uniform_real_distribution<float> g_perturb_axis_dist(
    0.f, static_cast<float>(hairrecon::PI));

template <typename T>
void Concat(std::vector<T>& v1, const std::vector<T>& v2) {
  std::copy(v2.begin(), v2.end(), std::back_inserter(v1));
}

template <typename T>
std::vector<T> RemoveByFlag(std::vector<T>& src, const std::vector<bool> flag) {
  std::vector<T> dst;
  dst.reserve(src.size());
  for (auto i = 0; i < src.size(); i++) {
    if (flag[i]) {
      dst.emplace_back(src[i]);
    }
  }
  return dst;
}

struct Sample2D {
  Eigen::Vector2f pos;
  ugu::Line2d projected_line;
};

std::tuple<std::vector<Sample2D>, std::vector<std::vector<Sample2D>>>
Sample2DPoints(hairrecon::KeyframePtr kf, int x, int y,
               const hairrecon::LpmvsParams& params) {
  // Get 3D line of (x, y)
  auto& line3d_param = kf->line3d.at<ugu::Vec3f>(y, x);
  ugu::Line3d&& camera_l = hairrecon::GetLine3dInCameraCoord(
      kf->camera, x, y, line3d_param[0], line3d_param[1], line3d_param[2]);

  // Project line direction to 2D
  ugu::Line2d ref_image_l;
  kf->camera->Project(camera_l, &ref_image_l);

#if 0
  static int count = 0;
  std::cout << count << std::endl;
  std::cout << "camera_l " << std::endl << "d " << camera_l.d << std::endl << "a " << camera_l.a << std::endl << std::endl;
  std::cout << "ref_image_l " << ref_image_l.d
            << " " << ref_image_l.a << std::endl;
  cv::Mat3b ref_image_l_cv = cv::Mat3b::zeros(480, 640);
  cv::line(ref_image_l_cv, cv::Point(0, ref_image_l.GetY(0)),
           cv::Point(639, ref_image_l.GetY(639)), 255);
  cv::imwrite(std::to_string(count) + ".png", ref_image_l_cv);
  count++;
#endif

  // Above process is not numerically acurate
  // Ensure to pass (x, y)
  if (std::abs(ref_image_l.p1.x() - x) < std::abs(ref_image_l.p0.x() - x)) {
    ref_image_l.Set(ref_image_l.p0, {x, y});
  } else {
    ref_image_l.Set(ref_image_l.p1, {x, y});
  }

  // For this view, sample k points along with the projected line
  // Find 2 crossing points of th line and  a circle with sample_2d_r radius by
  // Quadratic formula
  auto a = 1 + ref_image_l.d * ref_image_l.d;
  auto b = -2 * x + 2 * ref_image_l.a * ref_image_l.d - 2 * y * ref_image_l.d;
  auto c = ref_image_l.a * ref_image_l.a -
           params.sample_2d_r * params.sample_2d_r + x * x -
           2 * y * ref_image_l.a + y * y;
  auto D = b * b - 4 * a * c;
  double sample2d_max_x, sample2d_min_x;
  if (D >= 0) {
    // Accept D == 0
    sample2d_max_x = (-b + std::sqrt(D)) / (2 * a);
    sample2d_min_x = (-b - std::sqrt(D)) / (2 * a);
  } else {
    // ugu::LOGE("D is not positive (a %f, b %f, c %f, D %f)\n", a, b, c, D);
    // ugu::LOGE("ref_image_l.a %f, ref_image_l.d %f)\n", ref_image_l.a,
    //          ref_image_l.d);

    // ugu::LOGI("(%d %d) -> (%f %f)\n", x, y, ref_image_l.GetX(y),
    //         ref_image_l.GetY(x));
    sample2d_max_x = x + params.sample_2d_r;
    sample2d_min_x = x - params.sample_2d_r;
  }
  sample2d_max_x =
      std::min(sample2d_max_x, static_cast<double>(kf->camera->width() - 1));
  auto y0 = ref_image_l.GetY(sample2d_max_x);
  if (y0 < 0) {
    sample2d_max_x = ref_image_l.GetX(0.0);
  } else if (kf->camera->height() - 1 < y0) {
    sample2d_max_x = ref_image_l.GetX(kf->camera->height() - 1);
  }

  sample2d_min_x = std::max(sample2d_min_x, 0.0);
  auto y1 = ref_image_l.GetY(sample2d_min_x);
  if (y1 < 0) {
    sample2d_min_x = ref_image_l.GetX(0.0);
  } else if (kf->camera->height() - 1 < y1) {
    sample2d_min_x = ref_image_l.GetX(kf->camera->height() - 1);
  }

  // sample2d_min_x == sample2d_max_x may happen image edge
  // But it is ok since just sample the same edge point..
  assert(0 <= sample2d_min_x && sample2d_min_x <= kf->camera->width() - 1 &&
         0 <= sample2d_max_x && sample2d_max_x <= kf->camera->width() - 1 &&
         sample2d_min_x <= sample2d_max_x);

  auto sample2d_step_x =
      (sample2d_max_x - sample2d_min_x) / (params.sample_2d_k - 1);
  std::vector<Sample2D> s0;  // Add this point
  Sample2D s0_s;
  s0_s.pos = Eigen::Vector2f(x, y);
  s0_s.projected_line = ref_image_l;
  s0.emplace_back(s0_s);

  // Sample k - 1 points
  for (int i = 0; i < params.sample_2d_k - 1; i++) {
    auto curr_x = i * sample2d_step_x + sample2d_min_x;
    auto curr_y = ref_image_l.GetY(curr_x);

    curr_x =
        std::clamp(curr_x, 0.0, static_cast<double>(kf->camera->width() - 1));
    curr_y =
        std::clamp(curr_y, 0.0, static_cast<double>(kf->camera->height() - 1));

    Sample2D s;
    s.pos = Eigen::Vector2f(curr_x, curr_y);
    s.projected_line = ref_image_l;
    s0.emplace_back(s);
  }

  std::vector<Eigen::Vector3f> s0_3d;
  for (int j = 0; j < params.sample_2d_k; j++) {
    const auto& s = s0[j];
    Eigen::Vector3f ray;
    kf->camera->ray_c(s.pos.x(), s.pos.y(), &ray);
    ugu::Line3d ray_line;
    ray_line.d = ray.cast<double>();
    ray_line.a.setZero();  // ray comes from origin

    // Get the 3D line and ray intersection point
    auto [ret, exact_p] = FindExactLineCrossingPoint(camera_l, ray_line);
    if (!ret) {
      // if det is too small, solve least squares
      // ugu::LOGW("det is too small. Solve Least-Squares.\n");
      std::vector<ugu::Line3d> lines{camera_l, ray_line};
      auto [lsq_p, error] = ugu::FindBestLineCrossingPointLeastSquares(lines);
      s0_3d.emplace_back(lsq_p.cast<float>());
      continue;
    }
    s0_3d.emplace_back(exact_p.cast<float>());
  }

  // For neighbor views, sample k points
  std::vector<std::vector<Sample2D>> neighbor_samples;
  std::vector<bool> inside_image(params.sample_2d_k, true);
  for (auto i = 0; i < kf->neighbor_kfs.size(); i++) {
    const auto& neighbor_kf = kf->neighbor_kfs[i];
    std::vector<Sample2D> neighbor_sample;

    Eigen::Affine3d pose_diff = neighbor_kf->camera->w2c() * kf->camera->c2w();

    // Project line to neighbor view
    ugu::Line3d neighbor_camera_l = pose_diff * camera_l;
    ugu::Line2d neighbor_image_l;
    neighbor_kf->camera->Project(neighbor_camera_l, &neighbor_image_l);

    // std::cout << "ref_image_l " << ref_image_l.d << " " << ref_image_l.a
    //           << std::endl;
    // std::cout << "neighbor_image_l " << neighbor_image_l.d << " "
    //           << neighbor_image_l.a << std::endl;

    for (int j = 0; j < params.sample_2d_k; j++) {
      auto& ref_camera_p = s0_3d[j];

      // Project point to neighbor view
      Eigen::Vector3f neighbor_camera_p =
          pose_diff.cast<float>() * ref_camera_p;
      Eigen::Vector2f neighbor_image_p;
      neighbor_kf->camera->Project(neighbor_camera_p, &neighbor_image_p);

      if (neighbor_image_p.x() < 0 ||
          neighbor_kf->camera->width() - 1 < neighbor_image_p.x() ||
          neighbor_image_p.y() < 0 ||
          neighbor_kf->camera->height() - 1 < neighbor_image_p.y()) {
        inside_image[j] = false;
      }
      Sample2D s;
      s.pos = neighbor_image_p;
      s.projected_line = neighbor_image_l;
      neighbor_sample.emplace_back(s);
    }

    neighbor_samples.emplace_back(neighbor_sample);
  }

  // Remove points not visible (not inside image -> unable to calculate cost)
  // from all neighbor views
  s0 = RemoveByFlag(s0, inside_image);
  for (auto i = 0; i < kf->neighbor_kfs.size(); i++) {
    auto& neighbor_sample = neighbor_samples[i];
    neighbor_sample = RemoveByFlag(neighbor_sample, inside_image);
  }

#if 0
  {
    static int count1 = 0;
    cv::Mat3b ref_image_l_cv = cv::Mat3b::zeros(480, 640);
    if (!s0.empty()) {
      cv::line(ref_image_l_cv, cv::Point(0, ref_image_l.GetY(0)),
               cv::Point(639, ref_image_l.GetY(639)), 255);
      for (auto& s : s0) {
        cv::circle(ref_image_l_cv, cv::Point(s.pos.x(), s.pos.y()), 2,
                   cv::Scalar(0, 0, 255), -1);
      }
      cv::imwrite(std::to_string(count1) + "_ref.png", ref_image_l_cv);
    }
    for (int k = 0; k < neighbor_samples.size(); k++) {
      auto& neighbor_sample = neighbor_samples[k];
      if (neighbor_sample.empty()) {
        continue;
      }
      cv::Mat3b ns = cv::Mat3b::zeros(480, 640);
      cv::line(ns, cv::Point(0, neighbor_sample[0].projected_line.GetY(0)),
               cv::Point(639, neighbor_sample[0].projected_line.GetY(639)),
               255);
      for (auto& s : neighbor_sample) {
        cv::circle(ns, cv::Point(s.pos.x(), s.pos.y()), 2,
                   cv::Scalar(0, 0, 255), -1);
      }
      cv::imwrite(std::to_string(count1) + "_ns_" + std::to_string(k) + ".png",
                  ns);
    }

    count1++;
  }
#endif

  return {std::move(s0), std::move(neighbor_samples)};
}

#ifdef UGU_USE_OPENCV
float BilinearInterpolation1f(float x, float y, const ugu::Image1f& image) {
  std::array<int, 2> pos_min = {{0, 0}};
  std::array<int, 2> pos_max = {{0, 0}};
  pos_min[0] = static_cast<int>(std::floor(x));
  pos_min[1] = static_cast<int>(std::floor(y));
  pos_max[0] = pos_min[0] + 1;
  pos_max[1] = pos_min[1] + 1;

  // really need these?
  if (pos_min[0] < 0.0f) {
    pos_min[0] = 0;
  }
  if (pos_min[1] < 0.0f) {
    pos_min[1] = 0;
  }

  if (image.cols <= pos_min[0]) {
    pos_min[0] = image.cols - 1;
  }
  if (image.rows <= pos_min[1]) {
    pos_min[1] = image.rows - 1;
  }

  if (image.cols <= pos_max[0]) {
    pos_max[0] = image.cols - 1;
  }
  if (image.rows <= pos_max[1]) {
    pos_max[1] = image.rows - 1;
  }

  float local_u = x - pos_min[0];
  float local_v = y - pos_min[1];

  // bilinear interpolation
  float colorf =
      (1.0f - local_u) * (1.0f - local_v) *
          image.template at<float>(pos_min[1], pos_min[0]) +
      local_u * (1.0f - local_v) *
          image.template at<float>(pos_min[1], pos_max[0]) +
      (1.0f - local_u) * local_v *
          image.template at<float>(pos_max[1], pos_min[0]) +
      local_u * local_v * image.template at<float>(pos_max[1], pos_max[0]);

  return colorf;
}
#endif

double GeometricCostPerView(hairrecon::KeyframeConstPtr kf,
                            const std::vector<Sample2D>& sample) {
  double cost = 0.0;
  double sum_confidence = 0.0;
  int valid_sample_num = 0;
  for (auto i = 0; i < sample.size(); i++) {
    auto s = sample[i];
    auto y = s.pos.y();
    auto x = s.pos.x();

    if (std::isnan(x) || std::isnan(y)) {
      continue;
    }
#ifdef UGU_USE_OPENCV
    const auto& c = BilinearInterpolation1f(x, y, kf->confidence);
#else
    const auto& c = ugu::BilinearInterpolation(x, y, 0, kf->confidence);
#endif
    //   kf->confidence.at<float>(static_cast<int>(y), static_cast<int>(x));
    if (c <= std::numeric_limits<float>::epsilon()) {
      // Skip if confidence is 0
      continue;
    }

    // Note: Angle should be radian
    // ToDo: Check angle defintion (radian, from_y_axis, clockwise, etc.)
    double detected_angle =
        kf->orientation2d.at<float>(static_cast<int>(y), static_cast<int>(x));
    double projected_angle = hairrecon::CalcAngle(s.projected_line.d, 1.0);

    // std::cout << "detected_angle " << detected_angle << std::endl;
    // std::cout << "projected_angle " << projected_angle << std::endl;
    // std::cout << s.projected_line.a << " " << s.projected_line.d <<
    // std::endl;

    double angle_diff =
        hairrecon::CalcMinAngleDist(detected_angle, projected_angle);

    cost += c * angle_diff;

    sum_confidence += c;
    valid_sample_num++;
  }

  if (valid_sample_num > 0) {
    cost /= sum_confidence;
  } else {
    // Set big value
    cost = hairrecon::MAX_COST;
  }

  return cost;
}

double GeometricCost(hairrecon::KeyframeConstPtr kf,
                     const std::vector<Sample2D>& s0,
                     const std::vector<std::vector<Sample2D>>& samples) {
  const auto& neighbor_kfs = kf->neighbor_kfs;
  double cost = 0.0;

  float gamma_s0 = static_cast<float>(samples.size());
  // s0
  auto cost0 = GeometricCostPerView(kf, s0);
  if (hairrecon::MAX_COST <= cost0) {
    cost0 = hairrecon::MAX_COST;
    return hairrecon::MAX_COST;
  }
  cost += gamma_s0 * cost0;
  float gamma = 1.f;
  // other samples
  for (auto i = 0; i < samples.size(); i++) {
    auto c = GeometricCostPerView(neighbor_kfs[i], samples[i]);
    if (hairrecon::MAX_COST <= c) {
      c = hairrecon::MAX_COST;
      return hairrecon::MAX_COST;
    }
    cost += gamma * c;
  }

  cost /= (gamma_s0 + gamma * samples.size());
  if (hairrecon::MAX_COST <= cost) {
    cost = hairrecon::MAX_COST;
  }
  return cost;
}

double NCC1D(const ugu::Image1f& gray0, const std::vector<Sample2D>& s0,
             const ugu::Image1f& gray1, const std::vector<Sample2D>& s1) {
  assert(s0.size() == s1.size());

  // ToDo: Bilinear interpolation
  double sum0 = std::numeric_limits<double>::epsilon();
  double sum1 = std::numeric_limits<double>::epsilon();
  double inner = std::numeric_limits<double>::epsilon();
  for (auto i = 0; i < s0.size(); i++) {
    int y0 = static_cast<int>(s0[i].pos.y());
    int x0 = static_cast<int>(s0[i].pos.x());

    if (std::isnan(s1[i].pos.x()) || std::isnan(s0[i].pos.y())) {
      return hairrecon::MAX_COST;
    }

    const auto& val0 = gray0.at<float>(y0, x0);
    sum0 += (val0 * val0 + std::numeric_limits<double>::epsilon());

    int y1 = static_cast<int>(s1[i].pos.y());
    int x1 = static_cast<int>(s1[i].pos.x());
    const auto& val1 = gray1.at<float>(y1, x1);
    sum1 += (val1 * val1 + std::numeric_limits<double>::epsilon());

    inner += (val0 * val1 + std::numeric_limits<double>::epsilon());
  }

  auto denom = std::sqrt(sum0 * sum1);
#if 0
  if (denom <= std::numeric_limits<double>::epsilon()) {
    if (sum0 <= std::numeric_limits<double>::epsilon() &&
        sum1 <= std::numeric_limits<double>::epsilon()) {
        // Both all pixels are 0
        // Return maximum value 1.0
        return 1.0;
    }
  }
#endif  // 0

  return inner / denom;
}

double IntensityCost(hairrecon::KeyframeConstPtr kf,
                     const std::vector<Sample2D>& s0,
                     const std::vector<std::vector<Sample2D>>& samples) {
  assert(kf->neighbor_kfs.size() > 0 &&
         kf->neighbor_kfs.size() == samples.size());

  double cost = 0;
  for (auto i = 0; i < kf->neighbor_kfs.size(); i++) {
    // Original equation in the paper.
    // But this is wrong since NCC (0~1) gets larger if 2 samples get similar
    // cost += NCC1D(kf->intensity, s0, kf->neighbor_kfs[i]->intensity,
    // samples[i]);

    // So I took 1 - NCC to make cost larger if 2 samples get different
    // Possible another implementation is inverse of NCC
    cost += (1.0 - NCC1D(kf->intensity, s0, kf->neighbor_kfs[i]->intensity,
                         samples[i]));
  }

  cost /= kf->neighbor_kfs.size();

  return cost;
}

double CalcCost(hairrecon::KeyframePtr kf, int x, int y,
                const hairrecon::LpmvsParams& params) {
  if (params.skip_low_confidence) {
    const auto& c = kf->confidence.at<float>(y, x);
    if (c <= std::numeric_limits<float>::epsilon()) {
      // Skip if confidence is 0
      return hairrecon::MAX_COST;
    }
  }

  auto&& [s0, samples] = Sample2DPoints(kf, x, y, params);

  auto geo_cost = GeometricCost(kf, s0, samples);
  auto intensity_cost = IntensityCost(kf, s0, samples);
  //  if ((geo_cost < hairrecon::MAX_COST && intensity_cost <
  //  hairrecon::MAX_COST)) {
  //   ugu::LOGI("geo %f, intensity %f\n", geo_cost, intensity_cost);
  //  }
  if (geo_cost >= hairrecon::MAX_COST ||
      intensity_cost >= hairrecon::MAX_COST) {
    return hairrecon::MAX_COST;
  }

  return (1.0 - params.alpha) * geo_cost + params.alpha * intensity_cost;
}

void CalcCost(hairrecon::KeyframePtr kf, const hairrecon::LpmvsParams& params) {
  auto h = kf->cost.rows;
  auto w = kf->cost.cols;

#pragma omp parallel for
  for (auto j = 0; j < h; j++) {
    for (auto i = 0; i < w; i++) {
      auto& c = kf->cost.at<float>(j, i);
      c = static_cast<float>(CalcCost(kf, i, j, params));
    }
  }
}

void UpdateWithHypotheses(int i, int j, hairrecon::KeyframePtr kf,
                          const hairrecon::LpmvsParams& params,
                          const std::vector<ugu::Vec3f>& hypotheses) {
  for (const auto& hypothesis : hypotheses) {
    const auto cost_now = kf->cost.at<float>(j, i);
    const auto param_now = kf->line3d.at<ugu::Vec3f>(j, i);

    kf->line3d.at<ugu::Vec3f>(j, i) = hypothesis;

    auto cost_new = CalcCost(kf, i, j, params);

    if (cost_new > cost_now) {
      // if cost increases, revert to original param;
      kf->line3d.at<ugu::Vec3f>(j, i) = param_now;
    } else {
      // if cost decreases, update cost
      kf->cost.at<float>(j, i) = static_cast<float>(cost_new);
    }
  }
}

void InitGlobalDistributions(const hairrecon::LpmvsParams& params) {
  g_depth_dist =
      std::uniform_real_distribution<float>(params.min_depth, params.max_depth);

  g_perturb_depth_dist = std::uniform_real_distribution<float>(
      -params.perturbation_depth, params.perturbation_depth);
  g_perturb_phi_dist = std::uniform_real_distribution<float>(
      -params.perturbation_direc_phi_rad, params.perturbation_direc_phi_rad);
  g_perturb_theta_dist = std::uniform_real_distribution<float>(
      -params.perturbation_direc_theta_rad,
      params.perturbation_direc_theta_rad);
  g_perturb_axis_dist = std::uniform_real_distribution<float>(
      -params.perturbation_direc_axis_rad, params.perturbation_direc_axis_rad);
}

Eigen::Vector3d RandomSampleDirec() {
  // Galliani, Silvano, Katrin Lasinger, and Konrad Schindler. "Massively
  // parallel multiview stereopsis by surface normal diffusion." Proceedings of
  // the IEEE International Conference on Computer Vision. 2015.
  // 4. Multi-view Extension, 4.1. Parameterization in scene space,
  // Initialization

  // G. Marsaglia. Choosing a point from the surface of a sphere. Annals of
  // Mathematical Statistics, 43(2) : 645–646, 1972.

  double q1 = g_uniform_for_direc_dist(g_engine);
  double q2 = g_uniform_for_direc_dist(g_engine);
  double S = q1 * q1 + q2 * q2;
  while (S >= 1.0) {
    q1 = g_uniform_for_direc_dist(g_engine);
    q2 = g_uniform_for_direc_dist(g_engine);
    S = q1 * q1 + q2 * q2;
  }

  double s = std::sqrt(1 - S);
  Eigen::Vector3d direc(1.0 - 2.0 * S, 2 * q1 * s, 2 * q2 * s);

  // todo: limit direc

  return direc;
}

void RandomSampleThetaPhi(double& theta, double& phi, bool limit_phi = true) {
  Eigen::Vector3d direc = RandomSampleDirec();

  theta = std::acos(direc.z());
  phi = std::atan2(direc.y(), direc.x());

  if (limit_phi && phi > hairrecon::PI) {
    phi -= hairrecon::PI;
  }
}

#if 0
				  ugu::Vec3f RandomSampleLineParam(float x, float y,
                                        const ugu::LinePinholeCamera& camera) {
  ugu::Line3d l3d;
  l3d.d = RandomSampleDirec();
  Eigen::Vector3f p;
  auto d = g_depth_dist(g_engine);
  Eigen::Vector2f image_p{x, y};
  camera.Unproject(image_p, d, &p);
  l3d.a = p.cast<double>();
  ugu::Vec3f param;
  double depth, theta, phi;
  hairrecon::GetLineParam(l3d, depth, theta, phi);

  param[0] = static_cast<float>(depth);
  param[1] = static_cast<float>(theta);
  param[2] = static_cast<float>(phi);

  return std::move(param);
}

#endif  // 0

ugu::Vec3f RandomSampleLineParam() {
  double theta, phi;
  RandomSampleThetaPhi(theta, phi);

  ugu::Vec3b param;
  param[0] = g_depth_dist(g_engine);
  param[1] = static_cast<float>(theta);
  param[2] = static_cast<float>(phi);

  return std::move(param);
}

bool RandomInitializationUniform(ugu::Image3f& line3d, int w, int h,
                                 const hairrecon::LpmvsParams& params) {
  line3d = ugu::Image3f::zeros(h, w);

  // Initialize line paramters
  for (auto j = 0; j < h; j++) {
    for (auto i = 0; i < w; i++) {
      auto& l = line3d.at<ugu::Vec3f>(j, i);
      l = RandomSampleLineParam();
    }
  }

  return true;
}

void Sample3DLineFrom2DLine(const ugu::Line2d& l2d, int x, int y, double d1,
                            double d2, const ugu::LinePinholeCamera& camera,
                            ugu::Line3d& l3d) {
  const double large_offset =
      1000.0 * static_cast<double>(camera.width() + camera.height());

  Eigen::Vector2f a2d{x, y};  // current posiiton
  // Add large offset to alleviate numerical error
  auto sample_x = static_cast<double>(x + large_offset);
  Eigen::Vector2f b2d{sample_x,
                      l2d.GetY(sample_x)};  // moved point by the 2d line

  // Make two 3d points
  Eigen::Vector3f a3d, b3d;
  camera.Unproject(a2d, d1, &a3d);
  camera.Unproject(b2d, d2, &b3d);

  // Make 3d line connecting the points
  l3d.Set(a3d.cast<double>(), b3d.cast<double>());

  return;
}

bool RandomInitializationAlignOrien(ugu::Image3f& line3d, int w, int h,
                                    const hairrecon::LpmvsParams& params,
                                    const ugu::Image1f& orien,
                                    const ugu::Image1f& confidence,
                                    const ugu::LinePinholeCamera& camera) {
  line3d = ugu::Image3f::zeros(h, w);

  const double large_offset = 1000.0 * static_cast<double>(w + h);

  // Initialize line paramters
  for (auto j = 0; j < h; j++) {
    for (auto i = 0; i < w; i++) {
      const auto& c = confidence.at<float>(j, i);
      if (c < std::numeric_limits<float>::epsilon()) {
        // if confidence is too small, sample uniformly
        auto& l = line3d.at<ugu::Vec3f>(j, i);
        l = RandomSampleLineParam();
        continue;
      }

      // Sample a 2d point aligned with orien
      const auto& rad = orien.at<float>(j, i);
      auto l2d = hairrecon::Line2FomAngle(static_cast<double>(i),
                                          static_cast<double>(j),
                                          static_cast<double>(rad));

      // Sample two depth values
      float ad = g_depth_dist(g_engine);
      float bd = g_depth_dist(g_engine);

      // Sample 3d line from 2d line
      ugu::Line3d l3d;
      Sample3DLineFrom2DLine(l2d, i, j, ad, bd, camera, l3d);

      // Convert to minimum three parameters
      auto& l = line3d.at<ugu::Vec3f>(j, i);

      double depth, theta, phi;
      hairrecon::GetLineParam(l3d, depth, theta, phi);

      l[0] = static_cast<float>(depth);
      l[1] = static_cast<float>(theta);
      l[2] = static_cast<float>(phi);
    }
  }

  return true;
}

ugu::Vec3f SpatialPropagation(int nowx, int nowy, int fromx, int fromy,
                              const hairrecon::KeyframePtr kf,
                              const hairrecon::LpmvsParams& params) {
  Eigen::Vector3f ray_c;
  kf->camera->ray_c(nowx, nowy, &ray_c);
  ugu::Line3d l0;
  l0.d = ray_c.cast<double>();
  l0.a.setZero();

  const auto& l1_param = kf->line3d.at<ugu::Vec3f>(fromy, fromx);
  ugu::Line3d l1 = hairrecon::GetLine3dInCameraCoord(
      kf->camera, fromx, fromy, l1_param[0], l1_param[1], l1_param[2]);

  // Find the closest point
  double t, l_t;
  Eigen::Vector3d p0, p1;
  auto dist = l0.Dist(l1, &t, &p0, &l_t, &p1);

  // Generate a new hypothesis line
  ugu::Line3d new_l;
  new_l.a = p0;
  new_l.d = l1.d;

  ugu::Vec3d param_new;
  hairrecon::GetLineParam(new_l, param_new[0], param_new[1], param_new[2]);

  return ugu::Vec3f(param_new[0], param_new[1], param_new[2]);
}

ugu::Vec3f RandomSearch(int x, int y, hairrecon::KeyframePtr kf,
                        const hairrecon::LpmvsParams& params) {
  ugu::Vec3f param_new = RandomSampleLineParam();
  return param_new;
}

bool RandomSampleThetaPhiAlignOrien(int x, int y, hairrecon::KeyframePtr kf,
                                    const hairrecon::LpmvsParams& params,
                                    double depth, double& theta, double& phi) {
  auto l3d =
      hairrecon::GetLine3dInCameraCoord(kf->camera, x, y, depth, theta, phi);
#if 0
				
  // Make a rotation axis that is orthogonal to line direction
  const Eigen::Vector3d base(0, 0, 1);
  const Eigen::Vector3d axis = line3d.d.cross(base).normalized();

  // Apply angle pertubation to the axis
  // This keeps projected 2d angle
  double rad = g_perturb_axis_dist(g_engine);
  line3d.d = Eigen::AngleAxisd(rad, axis) * line3d.d;
#endif  // 0

  ugu::Line2d l2d;
  kf->camera->Project(l3d, &l2d);

  double tmp_depth = g_depth_dist(g_engine);
  Sample3DLineFrom2DLine(l2d, x, y, depth, tmp_depth, *kf->camera, l3d);

  double d = depth;
  // Recover representation
  // Note that do not touch depth
  hairrecon::GetLineParam(l3d, d, theta, phi);

  return true;
}

ugu::Vec3f RandomSearchDirec(int x, int y, hairrecon::KeyframePtr kf,
                             const hairrecon::LpmvsParams& params) {
  const auto& param_now = kf->line3d.at<ugu::Vec3f>(y, x);

  ugu::Vec3f param_new = param_now;
  double theta = -1;
  double phi = -1;
  double depth = param_now[0];
  if (params.random_search_type == hairrecon::RandomInitType::kUniform) {
    RandomSampleThetaPhi(theta, phi);
  } else if (params.random_search_type ==
             hairrecon::RandomInitType::kAlignOrien) {
    RandomSampleThetaPhiAlignOrien(x, y, kf, params, depth, theta, phi);
  } else {
    throw std::exception("");
  }
  param_new[1] = static_cast<float>(theta);
  param_new[2] = static_cast<float>(phi);

  return param_new;
}

ugu::Vec3f RandomSearchDepth(int x, int y, hairrecon::KeyframePtr kf,
                             const hairrecon::LpmvsParams& params) {
  auto param_now = kf->line3d.at<ugu::Vec3f>(y, x);

  ugu::Vec3f param_new = param_new;
  param_new[0] = g_depth_dist(g_engine);

  return param_new;
}

ugu::Vec3f PerturbationSearchDirec(int x, int y, hairrecon::KeyframePtr kf,
                                   const hairrecon::LpmvsParams& params) {
  const auto& param_now = kf->line3d.at<ugu::Vec3f>(y, x);

  ugu::Vec3f param_new = param_now;
  param_new[1] = std::clamp(param_new[1] + g_perturb_theta_dist(g_engine), 0.0f,
                            static_cast<float>(hairrecon::PI));
  param_new[2] = std::clamp(param_new[2] + g_perturb_phi_dist(g_engine), 0.0f,
                            static_cast<float>(hairrecon::PI));

  return param_new;
}

ugu::Vec3f PerturbationSearchDepth(int x, int y, hairrecon::KeyframePtr kf,
                                   const hairrecon::LpmvsParams& params) {
  // auto cost_now = kf->cost.at<float>(y, x);
  auto param_now = kf->line3d.at<ugu::Vec3f>(y, x);

  ugu::Vec3f param_new = param_now;
  param_new[0] = std::clamp(param_new[0] + g_perturb_depth_dist(g_engine),
                            params.min_depth, params.max_depth);
  return param_new;
}

// Partial random initializaiton and perturbation desribed in colmap paper
// Schönberger, Johannes L., et al. "Pixelwise view selection for unstructured
// multi-view stereo." European Conference on Computer Vision. Springer, Cham,
// 2016.
std::vector<ugu::Vec3f> ColmapSearch(int x, int y, hairrecon::KeyframePtr kf,
                                     const hairrecon::LpmvsParams& params) {
  std::vector<ugu::Vec3f> hypotheses;

  hypotheses.push_back(RandomSearchDirec(x, y, kf, params));

  // for (int i = 0; i < 5; i++) {
  hypotheses.push_back(RandomSearchDepth(x, y, kf, params));
  //}

  hypotheses.push_back(PerturbationSearchDirec(x, y, kf, params));
  // for (int i = 0; i < 5; i++) {
  hypotheses.push_back(PerturbationSearchDepth(x, y, kf, params));
  //}

  return hypotheses;
}

bool LineBasedPatchMatchMVSBodyFromUpperLeft(
    hairrecon::KeyframePtr kf, const hairrecon::LpmvsParams& params, int iter) {
  auto h = kf->camera->height();
  auto w = kf->camera->width();
  auto r = static_cast<int>(std::ceil(params.sample_2d_r));
  for (int j = r + 1; j < h - r; j++) {
    if (!params.debug_dir.empty() && j % params.debug_row_interval == 0) {
      hairrecon::DebugDump(params.debug_dir,
                           std::to_string(iter) + "_" + std::to_string(j) + "_",
                           kf);
    }
    for (int i = r + 1; i < w - r; i++) {
      std::vector<ugu::Vec3f> hypotheses;
      if (params.spatial_propagation) {
        hypotheses.push_back(SpatialPropagation(i, j, i - 1, j, kf, params));
        hypotheses.push_back(SpatialPropagation(i, j, i, j - 1, kf, params));
      }
      if (params.random_search) {
        hypotheses.push_back(RandomSearch(i, j, kf, params));
      }
      if (params.colmap_search) {
        const auto h = ColmapSearch(i, j, kf, params);
        Concat(hypotheses, h);
      }

      UpdateWithHypotheses(i, j, kf, params, hypotheses);
    }
  }
  return true;
}

bool LineBasedPatchMatchMVSBodyFromLowerRight(
    hairrecon::KeyframePtr kf, const hairrecon::LpmvsParams& params, int iter) {
  auto h = kf->camera->height();
  auto w = kf->camera->width();
  auto r = static_cast<int>(std::ceil(params.sample_2d_r));
  for (int j = h - 2 - r; j >= r; j--) {
    if (!params.debug_dir.empty() && j % params.debug_row_interval == 0) {
      hairrecon::DebugDump(params.debug_dir,
                           std::to_string(iter) + "_" + std::to_string(j) + "_",
                           kf);
    }

    for (int i = w - 2 - r; i >= r; i--) {
      std::vector<ugu::Vec3f> hypotheses;

      if (params.spatial_propagation) {
        hypotheses.push_back(SpatialPropagation(i, j, i + 1, j, kf, params));
        hypotheses.push_back(SpatialPropagation(i, j, i, j + 1, kf, params));
      }
      if (params.random_search) {
        hypotheses.push_back(RandomSearch(i, j, kf, params));
      }
      if (params.colmap_search) {
        const auto h = ColmapSearch(i, j, kf, params);
        Concat(hypotheses, h);
      }
      UpdateWithHypotheses(i, j, kf, params, hypotheses);
    }
  }
  return true;
}

bool LineBasedPatchMatchMVSBodyRedBlack(hairrecon::KeyframePtr kf,
                                        const hairrecon::LpmvsParams& params,
                                        int iter, bool is_red_turn) {
  auto h = kf->camera->height();
  auto w = kf->camera->width();
  auto r = static_cast<int>(std::ceil(params.sample_2d_r));
  const int red_black_r = 5;
  r = std::max(r, red_black_r);

  // Red pixel  : 1. even row and odd col, 2. odd row and even col
  // Black pixel: 1. even row and even col, 2. odd col and odd col

#pragma omp parallel for
  for (int j = r; j < h - r; j++) {
    if (!params.debug_dir.empty() && j % params.debug_row_interval == 0) {
      hairrecon::DebugDump(params.debug_dir,
                           std::to_string(iter) + "_" + std::to_string(j) + "_",
                           kf);
    }

    bool even_row = j % 2 == 0;
    int original_start_col = r;
    bool is_original_start_even = original_start_col % 2 == 0;
    bool is_start_col_red = false;
    if ((even_row && !is_original_start_even) ||
        (!even_row && is_original_start_even)) {
      is_start_col_red = true;
    }
    int start_col = original_start_col;
    if ((is_red_turn && !is_start_col_red) ||
        (!is_red_turn && is_start_col_red)) {
      start_col++;
    }

    for (int i = start_col; i < w - r; i += 2) {
      std::vector<ugu::Vec3f> hypotheses;
      if (params.spatial_propagation) {
        // Faster pattern
        hypotheses.push_back(SpatialPropagation(i, j, i - 1, j, kf, params));
        hypotheses.push_back(SpatialPropagation(i, j, i, j - 1, kf, params));
        hypotheses.push_back(SpatialPropagation(i, j, i - 5, j, kf, params));
        hypotheses.push_back(SpatialPropagation(i, j, i, j - 5, kf, params));
        hypotheses.push_back(SpatialPropagation(i, j, i + 1, j, kf, params));
        hypotheses.push_back(SpatialPropagation(i, j, i, j + 1, kf, params));
        hypotheses.push_back(SpatialPropagation(i, j, i + 5, j, kf, params));
        hypotheses.push_back(SpatialPropagation(i, j, i, j + 5, kf, params));

        if (!params.red_black_fast) {
          // Additional complate pattern
          hypotheses.push_back(SpatialPropagation(i, j, i - 3, j, kf, params));
          hypotheses.push_back(
              SpatialPropagation(i, j, i - 2, j - 1, kf, params));
          hypotheses.push_back(
              SpatialPropagation(i, j, i - 1, j - 2, kf, params));
          hypotheses.push_back(SpatialPropagation(i, j, i, j - 3, kf, params));
          hypotheses.push_back(
              SpatialPropagation(i, j, i + 1, j - 2, kf, params));
          hypotheses.push_back(
              SpatialPropagation(i, j, i + 2, j - 1, kf, params));
          hypotheses.push_back(SpatialPropagation(i, j, i + 3, j, kf, params));

          hypotheses.push_back(
              SpatialPropagation(i, j, i - 2, j + 1, kf, params));
          hypotheses.push_back(
              SpatialPropagation(i, j, i - 1, j + 2, kf, params));
          hypotheses.push_back(SpatialPropagation(i, j, i, j + 3, kf, params));
          hypotheses.push_back(
              SpatialPropagation(i, j, i + 1, j + 2, kf, params));
          hypotheses.push_back(
              SpatialPropagation(i, j, i + 2, j + 1, kf, params));
        }
      }
      if (params.random_search) {
        hypotheses.push_back(RandomSearch(i, j, kf, params));
      }
      if (params.colmap_search) {
        const auto h = ColmapSearch(i, j, kf, params);
        Concat(hypotheses, h);
      }

      UpdateWithHypotheses(i, j, kf, params, hypotheses);
    }
  }
  return true;
}

bool LineBasedPatchMatchMVSBodyRed(hairrecon::KeyframePtr kf,
                                   const hairrecon::LpmvsParams& params,
                                   int iter) {
  return LineBasedPatchMatchMVSBodyRedBlack(kf, params, iter, true);
}

bool LineBasedPatchMatchMVSBodyBlack(hairrecon::KeyframePtr kf,
                                     const hairrecon::LpmvsParams& params,
                                     int iter) {
  return LineBasedPatchMatchMVSBodyRedBlack(kf, params, iter, false);
}

}  // namespace

namespace hairrecon {

ugu::Line3d GetLine3dInCameraCoord(
    std::shared_ptr<const ugu::LinePinholeCamera> camera, double x, double y,
    double depth, double theta, double phi) {
  Eigen::Vector3f image_p, camera_p;
  image_p.x() = static_cast<float>(x);
  image_p.y() = static_cast<float>(y);
  image_p.z() = static_cast<float>(depth);
  camera->Unproject(image_p, &camera_p);

  ugu::Line3d line3d;

  // https://en.wikipedia.org/wiki/Spherical_coordinate_system
  line3d.d.x() = std::sin(theta) * std::cos(phi);
  line3d.d.y() = std::sin(theta) * std::sin(phi);
  line3d.d.z() = std::cos(theta);

  line3d.a = camera_p.cast<double>();

  return std::move(line3d);
}

void GetLineParam(const ugu::Line3d& camera_line3d, double& depth,
                  double& theta, double& phi) {
  depth = camera_line3d.a[2];

  theta = std::acos(camera_line3d.d[2]);  // 0-PI
  phi = hairrecon::CalcAngle(camera_line3d.d[1], camera_line3d.d[0], 0.0, 0.0,
                             false, false);  // 0-PI
}

bool LpmvsInitializeRandomEngine(const LpmvsParams& params) {
  g_engine = std::default_random_engine(params.random_seed);
  return true;
}

bool LineBasedPatchMatchMVS(KeyframePtr kf, const LpmvsParams& params) {
  ugu::Timer timer, whole_timer;
  whole_timer.Start();
  // Initialize line parameter
  timer.Start();

  if (kf->neighbor_kfs.size() < 1) {
    return true;
  }

  InitGlobalDistributions(params);
  if (params.random_init_type == RandomInitType::kUniform) {
    RandomInitializationUniform(kf->line3d, kf->intensity.cols,
                                kf->intensity.rows, params);
  } else if (params.random_init_type == RandomInitType::kAlignOrien) {
    RandomInitializationAlignOrien(
        kf->line3d, kf->intensity.cols, kf->intensity.rows, params,
        kf->orientation2d, kf->confidence, *kf->camera);
  }

  timer.End();
  ugu::LOGI("RandomInitialization %fms\n", timer.elapsed_msec());
  timer.Start();
  // Initialize cost
  kf->cost = ugu::Image1f::zeros(kf->intensity.rows, kf->intensity.cols);
  CalcCost(kf, params);
  timer.End();
  ugu::LOGI("Initialize cost %fms\n", timer.elapsed_msec());

  {
    // Magic
    // Ensure InitRayTable
    Eigen::Vector3f tmp;
    kf->camera->ray_c(0, 0, &tmp);
    for (const auto& nkf : kf->neighbor_kfs) {
      nkf->camera->ray_c(0, 0, &tmp);
    }
  }

  if (!params.debug_dir.empty()) {
    hairrecon::DebugDump(params.debug_dir, "init_", kf);
  }

  for (int iter = 0; iter < params.iter; iter++) {
    ugu::LOGI("iter %d ", iter);
    timer.Start();

    if (params.red_black_propagation) {
      LineBasedPatchMatchMVSBodyRed(kf, params, iter);
      LineBasedPatchMatchMVSBodyBlack(kf, params, iter);
    } else {
      if (params.alternately_reverse && iter % 2 == 1) {
        LineBasedPatchMatchMVSBodyFromLowerRight(kf, params, iter);
      } else {
        LineBasedPatchMatchMVSBodyFromUpperLeft(kf, params, iter);
      }
    }
    timer.End();
    ugu::LOGI("took %fms\n", timer.elapsed_msec());
  }

  whole_timer.End();
  ugu::LOGI("LineBasedPatchMatchMVS %fms\n", whole_timer.elapsed_msec());
  return true;
}

bool LineBasedPatchMatchMVS(std::vector<KeyframePtr>& kfs,
                            const LpmvsParams& params) {
  bool ret = true;
  for (auto& kf : kfs) {
    ret = LineBasedPatchMatchMVS(kf, params);
    if (!ret) {
      break;
    }
  }
  return ret;
}

bool LineFiltering(KeyframePtr kf, float pos_th, float angle_th,
                   int match_view_th) {
  // kf->filtered_mask = ugu::Image1b::zeros(kf->intensity.size());
  ugu::Image1i match_view_num = ugu::Image1i::zeros(kf->intensity.size());

  const int window = 3;
  const int hw = window / 2;

  auto h = kf->line3d.rows;
  auto w = kf->line3d.cols;
  for (int j = 0; j < h; j++) {
    for (int i = 0; i < w; i++) {
      if (kf->confidence.at<float>(j, i) <= 0.f) {
        continue;
      }

      const auto& val = kf->line3d.at<cv::Vec3f>(j, i);
      // Recover Line3d from minimum 3 parameters
      ugu::Line3d kf_camera_l =
          GetLine3dInCameraCoord(kf->camera, i, j, val[0], val[1], val[2]);
      // const auto& kf_camera_p = kf_camera_l.a;

      for (const auto& neighbor_kf : kf->neighbor_kfs) {
        // Convert to neighbor view
        Eigen::Affine3d pose_diff =
            neighbor_kf->camera->w2c() * kf->camera->c2w();
        ugu::Line3d kf_neighbor_camera_l = pose_diff * kf_camera_l;
        const Eigen::Vector3d& kf_neighbor_camera_p = kf_neighbor_camera_l.a;

        // Project to neighbor image space
        Eigen::Vector2f neighbor_image_p;
        neighbor_kf->camera->Project(kf_neighbor_camera_p.cast<float>(),
                                     &neighbor_image_p);

        if (neighbor_image_p.x() < 1 ||
            neighbor_kf->line3d.cols - 2 < neighbor_image_p.x() ||
            neighbor_image_p.y() < 1 ||
            neighbor_kf->line3d.rows - 2 < neighbor_image_p.y()) {
          // Outside of image
          continue;
        }

        if (!std::isnormal(neighbor_image_p.x()) ||
            !std::isnormal(neighbor_image_p.y())) {
          // Something wrong:
          // e.g. zero division with camera_p.z() == 0
          continue;
        }

        // Get corresponding line3d of the neighbor view
        int n_x_ = static_cast<int>(std::round(neighbor_image_p.x()));
        int n_y_ = static_cast<int>(std::round(neighbor_image_p.y()));

        bool updated = false;
        for (int jj = -hw; jj <= hw; jj++) {
          for (int ii = -hw; ii <= hw; ii++) {
            int n_x = n_x_ + ii;
            int n_y = n_y_ + jj;

            if (n_x < 1 || neighbor_kf->line3d.cols - 2 < n_x || n_y < 1 ||
                neighbor_kf->line3d.rows - 2 < n_y) {
              // Outside of image
              continue;
            }

            const auto& neighbor_cost = neighbor_kf->cost.at<float>(n_y, n_x);
            if (MAX_COST <= neighbor_cost) {
              continue;
            }
            const auto& neighbor_val =
                neighbor_kf->line3d.at<cv::Vec3f>(n_y, n_x);
            ugu::Line3d neighbor_camera_l =
                GetLine3dInCameraCoord(kf->camera, n_x, n_y, neighbor_val[0],
                                       neighbor_val[1], neighbor_val[2]);

            // Check consistency
            auto trans_diff =
                (kf_neighbor_camera_l.a - neighbor_camera_l.a).norm();
            auto angle_diff =
                std::acos(kf_neighbor_camera_l.d.dot(neighbor_camera_l.d));

            if (trans_diff < pos_th &&
                (0 > angle_th || angle_diff < angle_th)) {
              // Update count
              auto& match_num = match_view_num.at<int>(j, i);
              match_num++;
              updated = true;
            }
            if (updated) {
              break;
            }
          }
          if (updated) {
            break;
          }
        }
      }
    }
  }

  kf->filtered_mask = (match_view_th <= match_view_num) * 255;

  kf->filtered_cost = ugu::Image1f::ones(kf->cost.size()) * MAX_COST;
  kf->cost.copyTo(kf->filtered_cost, kf->filtered_mask);
  kf->filtered_line3d = ugu::Image3f::zeros(kf->line3d.size());
  kf->line3d.copyTo(kf->filtered_line3d, kf->filtered_mask);

  return true;
}

bool LineFiltering(std::vector<KeyframePtr>& kfs, float pos_th, float angle_th,
                   int match_view_th) {
  for (auto& kf : kfs) {
    LineFiltering(kf, pos_th, angle_th, match_view_th);
  }

  return true;
}

bool ColorizeLine3d(std::shared_ptr<const ugu::LinePinholeCamera>& camera,
                    const cv::Mat3f& line3d, cv::Mat3b& vis_line3d) {
  auto h = line3d.rows;
  auto w = line3d.cols;
  vis_line3d = cv::Mat3b::zeros(h, w);
  for (int j = 0; j < h; j++) {
    for (int i = 0; i < w; i++) {
      const auto& val = line3d.at<cv::Vec3f>(j, i);

      if (val[0] < std::numeric_limits<float>::epsilon() &&
          val[1] < std::numeric_limits<float>::epsilon() &&
          val[2] < std::numeric_limits<float>::epsilon()) {
        continue;
      }

      // Recover Line3d from minimum 3 parameters
      ugu::Line3d l3d =
          GetLine3dInCameraCoord(camera, i, j, val[0], val[1], val[2]);
      // Get Line2d by projection
      ugu::Line2d l2d;
      camera->Project(l3d, &l2d);
      auto rad = static_cast<float>(CalcAngle(l2d.d, 1.0));
      vis_line3d.at<cv::Vec3b>(j, i) = AngleToBgr(rad);
    }
  }

  return true;
}

bool Cost2Gray(const cv::Mat1f& cost, cv::Mat1b& vis_cost, float min_cost,
               float max_cost) {
  double minc, maxc;

  cv::Mat1f tmp;
  cost.copyTo(tmp);

  cv::Mat1b mask = tmp >= hairrecon::MAX_COST;

  if (min_cost < max_cost) {
    minc = min_cost;
    maxc = max_cost;
  } else {
    cv::minMaxLoc(tmp, &minc, &maxc);
  }

  tmp.setTo(static_cast<float>(minc), mask);
  cv::minMaxLoc(tmp, &minc, &maxc);

  // std::cout << minc << " " << maxc << std::endl;

  cv::Mat1f normalized = (tmp - minc) / (maxc - minc);

  normalized.convertTo(vis_cost, CV_8UC1, 255);

  vis_cost.setTo(255, mask);

  return true;
}

CostStats GetCostStats(const cv::Mat1f& cost, bool ignore_invalid) {
  CostStats stats;
  stats.data.assign(cost.begin(), cost.end());

  if (ignore_invalid) {
    auto result =
        std::remove_if(stats.data.begin(), stats.data.end(),
                       [](float x) { return x >= hairrecon::MAX_COST; });
    // Erase-remove idiom
    stats.data.erase(result, stats.data.end());
  }

  if (stats.data.size() < 1) {
    return stats;
  }

#if 0
  auto minmax = std::minmax_element(stats.data.begin(), stats.data.end());
  stats.min = *minmax.first;
  stats.max = *minmax.second;
#endif  // 0

  std::sort(stats.data.begin(), stats.data.end());
  stats.min = stats.data.front();
  stats.max = stats.data.back();

  stats.mean = std::accumulate(stats.data.begin(), stats.data.end(), 0.f) /
               stats.data.size();

  double sq_sum = std::inner_product(stats.data.begin(), stats.data.end(),
                                     stats.data.begin(), 0.0);
  stats.stdev = std::sqrt(sq_sum / stats.data.size() - stats.mean * stats.mean);

  if (stats.data.size() < 3) {
    stats.median = stats.mean;
  } else {
    if (stats.data.size() % 2 == 1) {
      stats.median = stats.data[stats.data.size() / 2];
    } else {
      stats.median = (stats.data[stats.data.size() / 2 - 1] +
                      stats.data[stats.data.size() / 2]) *
                     0.5f;
    }
  }

  stats.valid_ratio =
      static_cast<float>(stats.data.size()) / static_cast<float>(cost.total());

  return stats;
}

bool DebugDump(const std::string& dir, const std::string& prefix,
               KeyframePtr kf) {
  std::ofstream ofs(dir + "/" + prefix + "stats.txt");

  if (!kf->line3d.empty()) {
    ugu::WriteBinary(dir + "/" + prefix + "line.bin", kf->line3d);
    cv::Mat3b vis_line3d;
    hairrecon::ColorizeLine3d(kf->camera, kf->line3d, vis_line3d);
    cv::imwrite(dir + "/" + prefix + "line.png", vis_line3d);
  }

  if (!kf->cost.empty()) {
    ugu::WriteBinary(dir + "/" + prefix + "cost.bin", kf->cost);
    cv::Mat1b vis_cost;
    hairrecon::Cost2Gray(kf->cost, vis_cost);
    cv::imwrite(dir + "/" + prefix + "cost.png", vis_cost);

    auto stats = GetCostStats(kf->cost);
    ugu::LOGI("Cost Stats:\n%s\n", stats.ToString().c_str());
    ofs << stats.ToString();
  }

  if (!kf->filtered_line3d.empty()) {
    ugu::WriteBinary(dir + "/" + prefix + "line_filter.bin",
                     kf->filtered_line3d);
    cv::Mat3b vis_filtered_line3d;
    hairrecon::ColorizeLine3d(kf->camera, kf->filtered_line3d,
                              vis_filtered_line3d);
    cv::imwrite(dir + "/" + prefix + "line_filter.png", vis_filtered_line3d);
  }

  if (!kf->filtered_cost.empty()) {
    ugu::WriteBinary(dir + "/" + prefix + "cost_filter.bin", kf->filtered_cost);
    cv::Mat1b vis_filtered_cost;
    hairrecon::Cost2Gray(kf->filtered_cost, vis_filtered_cost);
    cv::imwrite(dir + "/" + prefix + "cost_filter.png", vis_filtered_cost);
    auto stats = GetCostStats(kf->filtered_cost);
    ugu::LOGI("Filtered Cost Stats:\n%s\n", stats.ToString().c_str());
    ofs << stats.ToString();
  }

  return true;
}

}  // namespace hairrecon