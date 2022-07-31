#pragma once

#include <cmath>
#include <type_traits>

#include "ugu/line.h"

namespace hairrecon {

static inline const double PI = 3.14159265358979323846;

template <typename T, typename std::enable_if<std::is_floating_point<T>::value,
                                              std::nullptr_t>::type = nullptr>
T CalcAngle(T y, T x, T center_y = 0, T center_x = 0, bool from_y_axis = true,
            bool clockwise = true) {
  T angle(0);
  T first, second;
  if (from_y_axis) {
    first = x - center_x;
    second = y - center_y;
  } else {
    first = y - center_y;
    second = x - center_x;
  }

  if ((std::abs(first) < std::numeric_limits<T>::epsilon()) &&
      (std::abs(second) < std::numeric_limits<T>::epsilon())) {
    angle = 0.0;
  } else {
    angle = static_cast<T>(std::atan2(first, second));
  }

  if (clockwise) {
    angle *= static_cast<T>(-1.0f);
  }

  angle += static_cast<T>(PI);

  if (angle >= PI) {
    angle -= static_cast<T>(PI);
  }

  return angle;
}

template <typename T, typename std::enable_if<std::is_floating_point<T>::value,
                                              std::nullptr_t>::type = nullptr>
T CalcMinAngleDist(T rad_a, T rad_b) {
  T diff = rad_a - rad_b;
  return std::min(
      {std::abs(diff), std::abs(T(diff - PI)), std::abs(T(diff + PI))});
}

template <typename T, typename std::enable_if<std::is_floating_point<T>::value,
                                              std::nullptr_t>::type = nullptr>
ugu::Line2<T> Line2FomAngle(T x, T y, T rad, bool from_y_axis = true,
                            bool clockwise = true) {
  ugu::Line2<T> l;
#ifdef _DEBUG
  T org_rad = rad;
#endif

  if (rad >= PI) {
    rad -= static_cast<T>(PI);
  }

  if (clockwise) {
    rad *= static_cast<T>(-1.0f);
  }

  T xb, yb;

  if (from_y_axis) {
    yb = T(1) + y;
    xb = std::tan(rad) + x;
  } else {
    xb = T(1) + x;
    yb = std::tan(rad) + y;
  }

  l.Set(x, y, xb, yb);

#ifdef _DEBUG
  {
    auto recomputed_rad = CalcAngle(y, x, yb, xb, from_y_axis, clockwise);
    if (hairrecon::CalcMinAngleDist(recomputed_rad, org_rad) > 0.001f) {
      ugu::LOGE("expected %f, actual %f\n", rad, recomputed_rad);
      assert(false);
    }
  }
#endif

  return l;
}

}  // namespace hairrecon