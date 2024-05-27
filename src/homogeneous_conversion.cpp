#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>

void homogeneousPointsConversion() {
  std::vector<cv::Point3d> homogeneousPoints(4, cv::Point3d(0, 0, 0));
  homogeneousPoints[0] = cv::Point3d(0, 0, 0);
  homogeneousPoints[1] = cv::Point3d(1, 1, 1);
  homogeneousPoints[2] = cv::Point3d(-1, -1, -1);
  homogeneousPoints[3] = cv::Point3d(2, 2, 2);
  std::vector<cv::Point2d> inhomogeneousPointsPoints(4);
  cv::convertPointsFromHomogeneous(homogeneousPoints,
                                   inhomogeneousPointsPoints);
}

int main() {
  homogeneousPointsConversion();
  return 0;
}
