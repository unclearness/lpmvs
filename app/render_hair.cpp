/*
 * Copyright (C) 2019, unclearness
 * All rights reserved.
 */

#include <stdio.h>

#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "opencv2/calib3d.hpp"
#include "ugu/renderer/rasterizer.h"
#include "ugu/renderer/raytracer.h"
#include "ugu/util/geom_util.h"
#include "ugu/util/image_util.h"
#include "ugu/util/io_util.h"
#include "ugu/util/rgbd_util.h"
#include "ugu/util/string_util.h"

using ugu::Camera;
using ugu::Depth2Gray;
using ugu::Depth2Mesh;
using ugu::Depth2PointCloud;
using ugu::FaceId2RandomColor;
using ugu::Image1b;
using ugu::Image1f;
using ugu::Image1i;
using ugu::Image1w;
using ugu::Image3b;
using ugu::Image3f;
using ugu::imwrite;
using ugu::Mesh;
using ugu::MeshStats;
using ugu::Normal2Color;
using ugu::PinholeCamera;
using ugu::Renderer;
using ugu::RendererOption;
using ugu::WriteFaceIdAsText;
using ugu::zfill;

namespace {

void PreparePoseAndName(const std::shared_ptr<Mesh> mesh,
                        std::vector<Eigen::Affine3d>& pose_list,
                        std::vector<std::string>& name_list) {
  MeshStats stats = mesh->stats();
  Eigen::Vector3f center = stats.center;
  Eigen::Vector3f eye;
  Eigen::Matrix4f c2w_mat;

  // translation offset is the largest edge length of bounding box * 1.5
  Eigen::Vector3f diff = stats.bb_max - stats.bb_min;
  float offset = std::max(diff[0], std::max(diff[1], diff[2])) * 1.0f;

  // from front
  eye = center;
  eye[2] += offset;
  ugu::c2w(eye, center, Eigen::Vector3f(0, 1, 0), &c2w_mat);
  pose_list.push_back(Eigen::Affine3d(c2w_mat.cast<double>()));
  name_list.push_back("front");

#if 0
				 // from back
  eye = center;
  eye[2] -= offset;
  ugu::c2w(eye, center, Eigen::Vector3f(0, 1, 0), &c2w_mat);
  pose_list.push_back(Eigen::Affine3d(c2w_mat.cast<double>()));
  name_list.push_back("back");

  // from right
  eye = center;
  eye[0] += offset;
  ugu::c2w(eye, center, Eigen::Vector3f(0, 1, 0), &c2w_mat);
  pose_list.push_back(Eigen::Affine3d(c2w_mat.cast<double>()));
  name_list.push_back("right");

  // from left
  eye = center;
  eye[0] -= offset;
  ugu::c2w(eye, center, Eigen::Vector3f(0, 1, 0), &c2w_mat);
  pose_list.push_back(Eigen::Affine3d(c2w_mat.cast<double>()));
  name_list.push_back("left");

  // from top
  eye = center;
  eye[1] -= offset;
  ugu::c2w(eye, center, Eigen::Vector3f(0, 0, 1), &c2w_mat);
  pose_list.push_back(Eigen::Affine3d(c2w_mat.cast<double>()));
  name_list.push_back("top");

  // from bottom
  eye = center;
  eye[1] += offset;
  ugu::c2w(eye, center, Eigen::Vector3f(0, 0, -1), &c2w_mat);
  pose_list.push_back(Eigen::Affine3d(c2w_mat.cast<double>()));
  name_list.push_back("bottom");
#endif  // 0
}

void Test(const std::string& out_dir, std::shared_ptr<Mesh> mesh,
          std::shared_ptr<Camera> camera, const Renderer& renderer,
          bool number_prefix = true) {
  // images
  Image3b color;
  Image1f depth;
  Image1w depthw;
  Image3f normal;
  Image1b mask;
  Image1i face_id;
  Image1b vis_depth;
  Image3b vis_normal;
  Image3b vis_face_id;
  Mesh view_mesh, view_point_cloud;
  const float kMaxConnectZDiff = 100.0f;

  // for pose output by tum format
  std::vector<Eigen::Affine3d> poses;

  std::vector<Eigen::Affine3d> pose_list;
  std::vector<std::string> name_list;
  // PreparePoseAndName(mesh, pose_list, name_list);

  MeshStats stats = mesh->stats();
  Eigen::Vector3f center = stats.center;
  Eigen::Vector3f eye;
  Eigen::Matrix4f c2w_mat;

  // translation offset is the largest edge length of bounding box * 1.5
  Eigen::Vector3f diff = stats.bb_max - stats.bb_min;
  float offset = std::max(diff[0], std::max(diff[1], diff[2])) * 1.0f;

  // from front
  eye = center;
  eye[2] += offset;
  ugu::c2w(eye, center, Eigen::Vector3f(0, 1, 0), &c2w_mat);
  pose_list.push_back(Eigen::Affine3d(c2w_mat.cast<double>()));
  name_list.push_back("front");

  int x_step = 1;
  int y_step = 1;
  bool gl_coord = false;
  std::string material_name = "Depth2Mesh_mat";
  bool with_vertex_color = true;
  ugu::Image3f mesh_point_cloud;
  ugu::Image3f mesh_normal;

  for (size_t i = 0; i < pose_list.size(); i++) {
    Eigen::Affine3d& c2w = pose_list[i];
    std::string prefix = name_list[i];
    if (number_prefix) {
      prefix = zfill(i);
    }

    camera->set_c2w(c2w);
    renderer.Render(&color, &depth, &normal, &mask, &face_id);
    imwrite(out_dir + prefix + "_color.png", color);
    imwrite(out_dir + prefix + "_mask.png", mask);
    ugu::ConvertTo(depth, &depthw);
    imwrite(out_dir + prefix + "_depth.png", depthw);
    Depth2Gray(depth, &vis_depth);
    imwrite(out_dir + prefix + "_vis_depth.png", vis_depth);
    Normal2Color(normal, &vis_normal);
    imwrite(out_dir + prefix + "_vis_normal.png", vis_normal);
    FaceId2RandomColor(face_id, &vis_face_id);
    imwrite(out_dir + prefix + "_vis_face_id.png", vis_face_id);
    WriteFaceIdAsText(face_id, out_dir + prefix + "_face_id.txt");

    double minDepth, maxDepth;
    cv::minMaxLoc(depth, &minDepth, &maxDepth);
    ugu::LOGI("mind: %f, maxd: %f\n", minDepth, maxDepth);

    Depth2Mesh(depth, color, *camera, &view_mesh, kMaxConnectZDiff, x_step,
               y_step, gl_coord, material_name, with_vertex_color,
               &mesh_point_cloud, &mesh_normal);

    Normal2Color(mesh_normal, &vis_normal);
    imwrite(out_dir + prefix + "_vis_mesh_normal.png", vis_normal);

    // Depth2PointCloud(depth, color, *camera, &view_point_cloud);
    view_point_cloud = ugu::Mesh(view_mesh);
    view_point_cloud.RemoveFaces();
    view_point_cloud.WritePly(out_dir + prefix + "_mesh.ply");
    view_mesh.WriteObj(out_dir, prefix + "_mesh");
    poses.push_back(camera->c2w());
  }

  // ugu::WriteTumFormat(poses, out_dir + "tumpose.txt");

  // For stereo test
  double baseline_step = 0.025f;  //- diff.maxCoeff() / 5;
  double angle_step = 5;
  auto vec = eye - center;
  double r = (eye - center).norm();

  for (size_t i = 1; i < 7; i++) {
    Eigen::Affine3d c2w = pose_list[0];
    std::string prefix = name_list[0];
    if (number_prefix) {
      // prefix = zfill(i);
      prefix = "00000";
    }
    prefix = "r_" + std::to_string(i) + "_" + prefix;

    // Add baseline
    // c2w = c2w * Eigen::Translation3d(baseline_step * i, 0.0, 0.0);

    Eigen::Translation3d t(center.cast<double>());
    Eigen::Translation3d nega_t(-center.cast<double>());

    Eigen::Affine3d rot;
    if (i < 4) {
      rot = t *
            (Eigen::AngleAxisd(ugu::radians(angle_step * i),
                               Eigen::Vector3d(0, 1, 0))) *
            nega_t;
    } else {
      rot = t *
            (Eigen::AngleAxisd(ugu::radians(angle_step * (i - 3)),
                               Eigen::Vector3d(-1, 0, 0))) *
            nega_t;
    }

    Eigen::Affine3d moved_c2w = rot * c2w;

    camera->set_c2w(moved_c2w);
    poses.push_back(camera->c2w());
    renderer.Render(&color, &depth, &normal, &mask, &face_id);
    imwrite(out_dir + prefix + "_color.png", color);
  }

  ugu::WriteTumFormat(poses, out_dir + "tumpose.txt");
}

void AlignMesh(std::shared_ptr<Mesh> mesh) {
  // move center as origin
  MeshStats stats = mesh->stats();
  mesh->Translate(-stats.center);

  // rotate 180 deg. around x_axis to align z:forward, y:down, x:right,
  Eigen::Matrix3f R =
      Eigen::AngleAxisf(ugu::radians(180.0f), Eigen::Vector3f::UnitX())
          .toRotationMatrix();
  mesh->Rotate(R);

  // recover original translation
  mesh->Translate(stats.center);
}
}  // namespace

int main(int argc, char* argv[]) {
  (void)argc;
  (void)argv;


  std::string data_dir = "../data/recon_points/";
  std::string srcobj_path = data_dir + "hair0202_single.obj";

  std::ifstream ifs(srcobj_path);
  if (!ifs.is_open()) {
    printf("Please put %s\n", srcobj_path.c_str());
    return -1;
  }

  std::string obj_path = data_dir + "hair0203.obj";
#if 1
  std::shared_ptr<Mesh> srcmesh = std::make_shared<Mesh>();
  srcmesh->LoadObj(srcobj_path, data_dir);
  std::default_random_engine engine;
  auto stats = srcmesh->stats();
  auto max_bb_len = (stats.bb_max - stats.bb_min).maxCoeff();
  // std::normal_distribution<float> trans_dist(0.f, max_bb_len * 0.2f);
  std::uniform_real_distribution<float> trans_dist(-max_bb_len * 0.5f,
                                                   max_bb_len * 0.5f);
  std::uniform_real_distribution<float> axis_dist;
  std::normal_distribution<float> angle_dist(0.f, CV_PI / 180 * 2);
  Mesh tmp0, tmp2;
  for (int i = 0; i < 200; i++) {
    Mesh tmp1(*srcmesh.get());
    Eigen::Vector3f trans(trans_dist(engine), 0,  // trans_dist(engine),
                          0);  // trans_dist(engine) * 0.001);

    cv::Vec3f rvec(axis_dist(engine), axis_dist(engine), axis_dist(engine));
    rvec = cv::normalize(rvec);
    rvec *= angle_dist(engine);
    cv::Mat cvR;
    cv::Rodrigues(rvec, cvR);

    Eigen::Matrix3f R;
    std::memcpy(R.data(), cv::Mat(cvR.t()).data, sizeof(float) * 9);

    R = Eigen::Matrix3f::Identity();

    tmp1.Transform(R, trans);

    ugu::MergeMeshes(tmp0, tmp1, &tmp2);
    tmp0 = Mesh(tmp2);
  }
  *srcmesh.get() = tmp0;

  srcmesh->WriteObj(data_dir, "hair0203");

#endif  // 0

  // load mesh
  std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>();
  mesh->LoadObj(obj_path, data_dir);

  mesh->CalcNormal();

  // original mesh with z:backward, y:up, x:right, like OpenGL
  // align z:forward, y:down, x:right
  // AlignMesh(mesh);

  // initialize renderer with diffuse texture color and lambertian shading
  RendererOption option;
  option.diffuse_color = ugu::DiffuseColor::kNone;
  option.diffuse_shading = ugu::DiffuseShading::kLambertian;

  std::unique_ptr<ugu::Renderer> renderer =
      std::make_unique<ugu::Raytracer>(option);

  // set mesh
  renderer->set_mesh(mesh);

  // prepare mesh for rendering (e.g. make BVH)
  renderer->PrepareMesh();

  // Make PinholeCamera
  // borrow KinectV1 intrinsics of Freiburg 1 RGB
  // https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
  float r = 3.0f;  // scale to smaller size from VGA
  int width = static_cast<int>(640 * r);
  int height = static_cast<int>(480 * r);
  Eigen::Vector2f principal_point(318.6f * r, 255.3f * r);
  Eigen::Vector2f focal_length(517.3f * r, 516.5f * r);
  std::shared_ptr<Camera> camera = std::make_shared<PinholeCamera>(
      width, height, Eigen::Affine3d::Identity(), principal_point,
      focal_length);

  // set camera
  renderer->set_camera(camera);

  // test
  Test(data_dir, mesh, camera, *renderer);

  return 0;
}