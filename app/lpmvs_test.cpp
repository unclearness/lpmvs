#include "lpmvs.h"

#include <iostream>
#include <memory>

#include "opencv2/highgui.hpp"
#include "orientation_2d.h"
#include "ugu/mesh.h"
#include "ugu/util/geom_util.h"
#include "ugu/util/io_util.h"

void EstimateHairOrientation2D(const std::string& data_dir,
                               std::vector<std::string>& data_names,
                               const std::string& ext) {
  for (auto& data_name : data_names) {
    auto data_path = data_dir + data_name + ext;
    cv::Mat1b gray = cv::imread(data_path, cv::ImreadModes::IMREAD_GRAYSCALE);

    hairrecon::EstimateHairOrientation2DParams params;
    params.confidence_percentile_th = 0.75f;
    params.angle_split = 180;

    params.debug = false;

    // gray = 255 - gray;

    cv::imwrite(data_name + "_" + "input.jpg", gray);

    int mark_len = gray.rows / 10;
    auto mark = hairrecon::OrientatnToBgrMark(mark_len);
    // cv::imwrite("mark.png", mark);

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

    cv::imwrite(data_dir + data_name + "_" + "orien.png", vis_orien);
    cv::imwrite(data_dir + data_name + "_" + "orien_gray.png", vis_orien_gray);
    cv::imwrite(data_dir + data_name + "_" + "orien_raw.png", vis_orien_raw);
    cv::imwrite(data_dir + data_name + "_" + "high_confidence_mask.png",
                output.high_confidence_mask);
    cv::imwrite(data_dir + data_name + "_" + "fg_mask.png", output.fg_mask);

    ugu::WriteBinary(data_dir + data_name + "_" + "orien.bin", output.orien);
    ugu::WriteBinary(data_dir + data_name + "_" + "confidence.bin",
                     output.confidence);

#if 1
    // Iterative refinement: Input confidence
    // This process significantly improves 1st result
    hairrecon::EstimateHairOrientation2D(output.confidence.clone(), params,
                                         output);
    hairrecon::OrientationToBgr(output.orien, vis_orien);
    hairrecon::OrientationToBgr(output.orien_raw, vis_orien_raw);
    mark.copyTo(vis_orien_raw(roi));
    mark.copyTo(vis_orien(roi));
    cv::imwrite(data_dir + data_name + "_" + "orien2.png", vis_orien);
    cv::imwrite(data_dir + data_name + "_" + "orien2_gray.png", vis_orien_gray);
    cv::imwrite(data_dir + data_name + "_" + "orien_raw2.png", vis_orien_raw);
    cv::imwrite(data_dir + data_name + "_" + "high_confidence_mask2.png",
                output.high_confidence_mask);
    cv::imwrite(data_dir + data_name + "_" + "fg_mask2.png", output.fg_mask);
#endif  // 0
  }
}

int main() {
  namespace hr = hairrecon;

  std::string data_dir = "../data/recon_points/";
  std::vector<std::string> data_names = {"00000_color",     "r_1_00000_color",
                                         "r_2_00000_color", "r_3_00000_color",
                                         "r_4_00000_color", "r_5_00000_color",
                                         "r_6_00000_color"};
  std::string ext = ".png";
  EstimateHairOrientation2D(data_dir, data_names, ext);

  std::string tum_path = data_dir + "tumpose.txt";
  std::vector<Eigen::Affine3d> poses;
  ugu::LoadTumFormat(tum_path, &poses);

  float r = 3.0f;  // scale to smaller size from VGA
  int width = static_cast<int>(640 * r);
  int height = static_cast<int>(480 * r);
  Eigen::Vector2f principal_point(318.6f * r, 255.3f * r);
  Eigen::Vector2f focal_length(517.3f * r, 516.5f * r);

  std::vector<hr::KeyframePtr> kfs;
  std::vector<std::shared_ptr<ugu::LinePinholeCamera>> cameras;
  for (auto i = 0; i < data_names.size(); i++) {
    hr::KeyframePtr kf = hr::Keyframe::Create();
    auto camera = std::make_shared<ugu::LinePinholeCamera>();

    camera->set_size(width, height);
    camera->set_focal_length(focal_length);
    camera->set_principal_point(principal_point);
    camera->set_c2w(poses[i]);
    kf->camera = camera;

    kf->neighbor_kfs.push_back(kf);

    auto gray = cv::imread(data_dir + data_names[i] + ext,
                           cv::ImreadModes::IMREAD_GRAYSCALE);
    gray.convertTo(kf->intensity, CV_32FC1);

    kf->orientation2d =
        ugu::Image1f::zeros(kf->intensity.rows, kf->intensity.cols);
    kf->confidence =
        ugu::Image1f::zeros(kf->intensity.rows, kf->intensity.cols);
    ugu::LoadBinary(data_dir + data_names[i] + "_" + "orien.bin",
                    kf->orientation2d);
    ugu::LoadBinary(data_dir + data_names[i] + "_" + "confidence.bin",
                    kf->confidence);

    kfs.push_back(kf);
    cameras.push_back(camera);
  }

  for (auto i = 0; i < data_names.size(); i++) {
    for (auto j = 0; j < data_names.size(); j++) {
      if (i == j) {
        continue;
      }

      kfs[i]->neighbor_kfs.push_back(kfs[j]);
    }
  }

  hairrecon::LpmvsParams params;
  // params.red_black_propagation = false;
  // params.iter = 3;
  params.max_depth = 1.70f;
  params.min_depth = 1.40f;

  for (auto i = 0; i < data_names.size(); i++) {
    hairrecon::LineBasedPatchMatchMVS(kfs[i], params);
    hairrecon::DebugDump(data_dir, data_names[i] + "_", kfs[i]);
  }


  std::vector<std::shared_ptr<ugu::Mesh>> orien_meshes;
  std::vector<std::shared_ptr<ugu::Mesh>> orien_points;

#pragma omp parallel for
  for (auto i = 0; i < data_names.size(); i++) {
    auto size = kfs[i]->intensity.size();
    auto kf = kfs[i];
    kf->line3d = cv::Mat3f::zeros(size);
    kf->cost = cv::Mat1f::zeros(size);
    ugu::LoadBinary(data_dir + data_names[i] + "_" + "line.bin", kf->line3d);
    ugu::LoadBinary(data_dir + data_names[i] + "_" + "cost.bin", kf->cost);

    cv::Mat3b vis_line3d;
    hairrecon::ColorizeLine3d(kf->camera, kf->line3d, vis_line3d);
    cv::imwrite(data_dir + data_names[i] + "_" + "line.png", vis_line3d);
    cv::Mat1b vis_cost;
    hairrecon::Cost2Gray(kf->cost, vis_cost);
    cv::imwrite(data_dir + data_names[i] + "_" + "cost.png", vis_cost);

    hairrecon::LineFiltering(kf, 0.001f, ugu::radians(10.0f), 2);
    hairrecon::DebugDump(data_dir, data_names[i] + "_", kf);
  }

  for (auto i = 0; i < data_names.size(); i++) {
    std::vector<float> costs;
    auto kf = kfs[i];
    for (int j = 0; j < kf->cost.rows; j++) {
      for (int i = 0; i < kf->cost.cols; i++) {
        auto c = kf->cost.at<float>(j, i);
        if (c < hairrecon::MAX_COST) {
          costs.push_back(c);
        }
      }
    }

    std::sort(costs.begin(), costs.end());
    float cost_th = costs[costs.size() / 2];
    std::cout << i << " cost_th " << cost_th << std::endl;
    cost_th = 10000;

    auto orienmesh = std::make_shared<ugu::Mesh>();
    auto orien_point = std::make_shared<ugu::Mesh>();
    std::vector<Eigen::Vector3f> vertices;
    std::vector<Eigen::Vector3i> indices;

    std::vector<Eigen::Vector3f> points_vertices;
    std::vector<Eigen::Vector3f> points_direcs;

    ugu::Image1f depth = ugu::Image1f::zeros(kf->cost.size());
    std::vector<Eigen::Vector3f> orien;
    for (int j = 0; j < kf->cost.rows; j++) {
      for (int i = 0; i < kf->cost.cols; i++) {
        if (kf->cost.at<float>(j, i) < cost_th) {
          auto l_param = kf->filtered_line3d.at<cv::Vec3f>(j, i);
          auto d = l_param[0];
          if (params.min_depth < d && d < params.max_depth) {
            depth.at<float>(j, i) = d;

            auto l_3d = hairrecon::GetLine3dInCameraCoord(
                kf->camera, i, j, d, l_param[1], l_param[2]);
            // Normal
            auto normal = l_3d.d;
            Eigen::Vector3d up(1, 0, 0);
            if (up.dot(normal) < 0.00001) {
              up = Eigen::Vector3d(0, 1, 0);
            }
            // Tangent
            auto tangent = normal.cross(up);
            // Binormal
            auto binormal = tangent.cross(normal);

            Eigen::Vector3f camera_p;
            kf->camera->Unproject({i, j}, d, &camera_p);
            auto v0 = camera_p;
            auto v1 =
                (camera_p.cast<double>() + binormal * 0.001).cast<float>();
            auto v2 = (camera_p.cast<double>() + tangent * 0.001).cast<float>();

            auto index = static_cast<int>(vertices.size()) / 3;
            indices.push_back({index * 3, index * 3 + 1, index * 3 + 2});
            vertices.push_back(v0);
            vertices.push_back(v1);
            vertices.push_back(v2);

#if 0
          Eigen::Vector3d v1_ = (v1 -  v0).cast<double>().normalized();
          Eigen::Vector3d v2_ = (v2 - v0).cast<double>().normalized();
          auto tmp = v1_.cross(v2_).normalized();
          std::cout << tmp << std::endl
                    << normal << std::endl
                    << normal << std::endl
                    << (tmp - normal).norm()
                    << std::endl;
#endif  // 0

            orien.push_back(l_3d.d.cast<float>());

            points_vertices.push_back(camera_p);
            points_direcs.push_back(normal.cast<float>());
          }
        }
      }
    }

    orienmesh->set_vertices(vertices);
    orienmesh->set_vertex_indices(indices);
    orienmesh->Transform(kf->camera->c2w().rotation().cast<float>(),
                         kf->camera->c2w().translation().cast<float>());
    orienmesh->WritePly("orien_" + std::to_string(i) + ".ply");

    orien_meshes.push_back(orienmesh);

    orien_point->set_vertices(points_vertices);
    orien_point->set_normals(points_direcs);
    orien_point->Transform(kf->camera->c2w().rotation().cast<float>(),
                           kf->camera->c2w().translation().cast<float>());
    orien_point->WritePly("points_" + std::to_string(i) + ".ply");

    orien_points.push_back(orien_point);
#if 0
				  ugu::Mesh depthmesh;
  ugu::Depth2PointCloud(depth, *kf->camera, &depthmesh);
  depthmesh.set_normals(orien);
  depthmesh.Transform(kf->camera->c2w().rotation().cast<float>(),
                      kf->camera->c2w().translation().cast<float>());
  depthmesh.WriteObj("./", "result");
#endif  // 0
  }
  ugu::Mesh merged;
  ugu::MergeMeshes(orien_meshes, &merged);

  merged.WritePly("orien_merged.ply");

  ugu::Mesh merged_points;
  ugu::MergeMeshes(orien_points, &merged_points);

  merged_points.WritePly("points_merged.ply");
}