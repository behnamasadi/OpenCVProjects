#include "csv.h"
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

template <typename T> std::vector<T> createEllipsoidInWorldCoordinate() {
  std::vector<T> objectPointsInWorldCoordinate;

  float X, Y, Z;

  float phiStepSize, thetaStepSize;
  phiStepSize = 0.2;
  thetaStepSize = 0.1;
  float a, b, c;
  a = 2;
  b = 3;
  c = 1.6;
  for (float phi = -M_PI; phi < M_PI; phi = phi + phiStepSize) {
    for (float theta = -M_PI / 2; theta < M_PI / 2;
         theta = theta + thetaStepSize) {
      X = a * cos(theta) * cos(phi);
      Y = b * cos(theta) * sin(phi);
      Z = c * sin(theta);
      objectPointsInWorldCoordinate.push_back({X, Y, Z});
    }
  }

  return objectPointsInWorldCoordinate;
}

cv::Mat createImage(double focalLength, int numberOfPixelInHeight,
                    int numberOfPixelInWidth,
                    std::vector<cv::Point2f> projectedPoints,
                    std::string fileName) {

  double row, col;
  int blue, green, red;
  blue = 255;
  green = 255;
  red = 255;

  cv::Mat cameraImage =
      cv::Mat::zeros(numberOfPixelInHeight, numberOfPixelInWidth, CV_8UC1);
  cv::line(cameraImage, cv::Point2d(numberOfPixelInWidth / 2, 0),
           cv::Point2d(numberOfPixelInWidth / 2, numberOfPixelInHeight),
           cv::Scalar(255, 255, 255));
  cv::line(cameraImage, cv::Point2d(0, numberOfPixelInHeight / 2),
           cv::Point2d(numberOfPixelInWidth, numberOfPixelInHeight / 2),
           cv::Scalar(255, 255, 255));
  for (std::size_t i = 0; i < projectedPoints.size(); i++) {
    col = int(projectedPoints.at(i).x);
    row = int(projectedPoints.at(i).y);
    // std::cout << row << "," << col << std::endl;
    if (int(row) < numberOfPixelInHeight && int(col) < numberOfPixelInWidth)
      // cameraImage.at<char>(int(row), int(col)) = char(255);
      // draw a circle at (U,V) with a radius of 20. Use green lines of width 2
      cv::circle(cameraImage, cv::Point(int(col), int(row)), 2,
                 cv::Scalar(blue, green, red), 3);

    else {
      std::cout << row << "," << col << "is out of image" << std::endl;
    }
  }

  cv::imwrite(fileName, cameraImage);
  return cameraImage;
}
cv::Mat rotationMatrixFromRollPitchYaw(double alpha, double beta,
                                       double gamma) {
  /*
      yaw:
          A yaw is a counterclockwise rotation of alpha about the  z-axis. The
      rotation matrix is given by

          R_z

          |cos(alpha) -sin(alpha) 0|
          |sin(alpha)   cos(alpha) 0|
          |    0            0     1|

      pitch:
          R_y
          A pitch is a counterclockwise rotation of  beta about the  y-axis. The
      rotation matrix is given by

          |cos(beta)  0   sin(beta)|
          |0          1       0    |
          |-sin(beta) 0   cos(beta)|

      roll:
          A roll is a counterclockwise rotation of  gamma about the  x-axis. The
      rotation matrix is given by
          R_x
          |1          0           0|
          |0 cos(gamma) -sin(gamma)|
          |0 sin(gamma)  cos(gamma)|
  */

  cv::Mat R_z = (cv::Mat_<double>(3, 3) << cos(alpha), -sin(alpha), 0,
                 sin(alpha), cos(alpha), 0, 0, 0, 1);

  cv::Mat R_y = (cv::Mat_<double>(3, 3) << cos(beta), 0, sin(beta), 0, 1, 0,
                 -sin(beta), 0, cos(beta));

  cv::Mat R_x = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, cos(gamma), -sin(gamma),
                 0, sin(gamma), cos(gamma));

  return R_z * R_y * R_x;
}

template <typename T>
std::vector<T> readPoints(std::string pathToPointFile = "../data/points.csv") {
  io::CSVReader<3> in(pathToPointFile);
  in.read_header(io::ignore_extra_column, "x", "y", "z");
  double x, y, z;
  std::vector<T> objectPoints;

  while (in.read_row(x, y, z)) {
    objectPoints.push_back(T(x, y, z));
  }

  std::cout << "OpenCV coordinate:\n" << std::endl;

  std::cout << "                 Z" << std::endl;
  std::cout << "                ▲" << std::endl;
  std::cout << "               /" << std::endl;
  std::cout << "              /" << std::endl;
  std::cout << "             /1 2 3 4     X" << std::endl;
  std::cout << "            |------------ ⯈" << std::endl;
  std::cout << "           1|" << std::endl;
  std::cout << "           2|" << std::endl;
  std::cout << "           3|" << std::endl;
  std::cout << "           4|" << std::endl;
  std::cout << "            | Y" << std::endl;
  std::cout << "            ⯆" << std::endl;

  std::cout << "\npoints in 3d world:\n" << std::endl;

  for (const auto &p : objectPoints)
    std::cout << p << std::endl;

  return objectPoints;
}

int main(int argc, char **argv) {

  ///////////////// camera intrinsic /////////////////
  unsigned int numberOfPixelInHeight, numberOfPixelInWidth;
  double heightOfSensor, widthOfSensor;
  // double focalLength = 0.1;
  double focalLength = 2.0;
  double mx, my, cx, cy;

  numberOfPixelInHeight = 480;
  numberOfPixelInWidth = 640;

  heightOfSensor = 10;
  widthOfSensor = 10;

  mx = (numberOfPixelInWidth) / widthOfSensor;
  my = (numberOfPixelInHeight) / heightOfSensor;

  cx = (numberOfPixelInWidth) / 2;
  cy = (numberOfPixelInHeight) / 2;

  cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << focalLength * mx, 0, cx, 0,
                          focalLength * my, cy, 0, 0, 1);

  std::cout << "camera intrinsic:\n" << cameraMatrix << std::endl;

  //  4, 5, 8, 12 or 14 elements
  double k1, k2, p1, p2, k3;

  //   k1 = 0.;
  //   k2 = 0.;
  //   p1 = 0.;
  //   p2 = 0.;
  //   k3 = 0.;

  k1 = 1.6076213815298762e-01;
  k2 = -1.2591517698167454e+00;
  p1 = 0.;
  p2 = 0.;
  k3 = 3.1119684914597876e+00;
  k3 = 0.0;

  cv::Mat distortionCoefficient =
      (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);

  std::cout << "camera distortion coefficient:\n"
            << distortionCoefficient << std::endl;

  double roll, pitch, yaw, tx, ty, tz;

  roll = -M_PI / 36;
  pitch = +M_PI / 36;
  yaw = -M_PI / 36;

  tx = -0.5;
  ty = 0.4;
  tz = -4.0;

  // R_w_c cameraRotation
  cv::Mat R_w_c = rotationMatrixFromRollPitchYaw(roll, pitch, yaw);

  // T_w_c cameraTranslation
  cv::Mat T_w_c = (cv::Mat_<double>(3, 1) << tx, ty, tz);

  std::vector<cv::Vec3f> objectPointsInWorldesCoordinate;

  //   objectPointsInWorldesCoordinate =
  //       readPoints<cv::Vec3f>("../../data/points.csv");

  objectPointsInWorldesCoordinate =
      createEllipsoidInWorldCoordinate<cv::Vec3f>();

  ///////////////// 3D points from world /////////////////

  std::vector<cv::Point2f> projectedPointsInCamera;

  cv::Mat R_c_w = R_w_c.t(); // Transpose (or inverse) of R_w_c
  cv::Mat T_c_w =
      -R_c_w * T_w_c; // Correct transformation of the translation vector

  std::cout << "====projecting 3D points into camera  using OpenCV===="
            << std::endl;

  cv::projectPoints(objectPointsInWorldesCoordinate, R_c_w, T_c_w, cameraMatrix,
                    distortionCoefficient, projectedPointsInCamera);

  std::cout << "projected point in camera" << std::endl;
  for (const auto p : projectedPointsInCamera)
    std::cout << "row: " << p.y << ","
              << " column: " << p.x << std::endl;

  std::cout << "====saving projected point into image====" << std::endl;

  std::string fileName = std::string("image_") + std::to_string(focalLength) +
                         std::string("_.png");

  cv::Mat img =
      createImage(focalLength, numberOfPixelInHeight, numberOfPixelInWidth,
                  projectedPointsInCamera, fileName);

  double fx = focalLength * mx;
  double fy = focalLength * my;
  for (const auto p : projectedPointsInCamera) {

    cv::Point2f pt_in(p.x, p.y);
    cv::Mat pts(1, 1, CV_32FC2, &pt_in);

    cv::Mat undist;
    cv::undistortPoints(pts, undist, cameraMatrix, distortionCoefficient);
    std::cout << "OpenCV undist:                  " << undist << std::endl;

    double x_sn = (p.x - cx) / fx;
    double y_sn = (p.y - cy) / fy;
    std::cout << "distorted normalized coordinates:" << x_sn << "," << y_sn
              << std::endl;
  }
}
