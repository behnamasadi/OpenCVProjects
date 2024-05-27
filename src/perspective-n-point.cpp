#include <iostream>
#include <vector>
#define EIGEN_RUNTIME_NO_MALLOC // Define this symbol to enable runtime tests
                                // for allocations
#include "opencv2/core/eigen.hpp"
#include <Eigen/Dense>

//#include "include/projection_matrix.hpp"
//#include "include/dlt.hpp"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

void projectFromCameraCoordinateToCameraPlane(
    Eigen::Matrix3Xd &pointsInCameraCoordinate,
    Eigen::Matrix3d &cameraIntrinsicMatrix,
    Eigen::Matrix2Xd &pointsInCameraPlane) {
  /*
  Intrinsic camera Matrix is something like this:

  focalLength  ===> unit is mm
  mx=  ==>unit is Pixel/mm
  U0=  ===> unit is Pixel

  my=-(numberOfPixelInHeight)/heightOfSensor ;
  U0=(numberOfPixelInHeight)/2 ;

  mx=(numberOfPixelInWidth)/heightOfSensor;
  V0=(numberOfPixelInWidth)/2;

  Gamma=0

  X,Y,Z,W are homogeneous position of point in camera coordinate

  ┌ ┐	 ┌                         	 ┐┌ ┐
  v`	 |f*mx		Gamma		V0	0||X|
  u`	=|0			f*my		U0	0||Y|
  w`	 |0			0			 1	0||Z|
  └ ┘	 └                         	 ┘└W┘

                                                                           ▲y
                                                                           |
                          ----►V					 |
                          |	 ____________________|_____________________
                          |	|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
                  U	▼	|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
                                  |__|__|__|__|__|__|__|__|__|__|__|__|__|__|
                                  |__|__|__|__|__|__|__|__|__|__|__|__|__|__|
                                  |__|__|__|__|__|__|__|__|__|__|__|__|__|__|____________►
  x
                                  |__|__|__|__|__|__|__|__|__|__|__|__|__|__|
                                  |__|__|__|__|__|__|__|__|__|__|__|__|__|__|
                                  |__|__|__|__|__|__|__|__|__|__|__|__|__|__|
                                  |__|__|__|__|__|__|__|__|__|__|__|__|__|__|
                                  |__|__|__|__|__|__|__|__|__|__|__|__|__|__|



  To project points from camera coordinate to camera plane:

  V`=X*f*mx + V0*Z
  U`=Y*f*my + U0*Z
  W`=Z

  U=U`/W` ,V=V`/W`


  V=f*mx*X/Z + V0
  U=f*my*Y/Z + U0

  (U,V) is the index of corresponding point in the image
  */

  Eigen::Matrix3Xd pointsInCameraPlaneHomogeneous(
      pointsInCameraCoordinate.rows(), pointsInCameraCoordinate.cols());

  pointsInCameraPlaneHomogeneous =
      cameraIntrinsicMatrix * pointsInCameraCoordinate;
  std::cout << "pointsInCameraPlaneHomogeneous" << std::endl;
  std::cout << pointsInCameraPlaneHomogeneous << std::endl;
  for (int i = 0; i < pointsInCameraPlaneHomogeneous.cols(); i++) {
    pointsInCameraPlaneHomogeneous(0, i) =
        pointsInCameraPlaneHomogeneous(0, i) /
        pointsInCameraPlaneHomogeneous(2, i);
    pointsInCameraPlaneHomogeneous(1, i) =
        pointsInCameraPlaneHomogeneous(1, i) /
        pointsInCameraPlaneHomogeneous(2, i);
  }
  pointsInCameraPlane = pointsInCameraPlaneHomogeneous.block(
      0, 0, pointsInCameraPlaneHomogeneous.rows() - 1,
      pointsInCameraPlaneHomogeneous.cols());
}

void transformPoints(Eigen::Matrix3Xd &points, Eigen::Matrix3d &rotationMatrix,
                     Eigen::Vector3d &translationVector,
                     Eigen::Matrix3Xd &transformedPoints) {
  transformedPoints = (rotationMatrix * points).colwise() + translationVector;
}

void projectFromCameraCoordinateToCameraPlane_Example() {

  /*
   OpenCV coordinate system
                    ▲
                   /
                  /
                Z/
                /
               /
  ------------------------------► X
              |
              |
              |
              |
            Y |
              ▼
  */

  int numberOfPixelInHeight, numberOfPixelInWidth;
  double heightOfSensor, widthOfSensor;
  double focalLength = 1.5;
  double mx, my, U0, V0;
  numberOfPixelInHeight = 600;
  numberOfPixelInWidth = 600;

  heightOfSensor = 10;
  widthOfSensor = 10;

  my = (numberOfPixelInHeight) / heightOfSensor;
  U0 = (numberOfPixelInHeight) / 2;

  mx = (numberOfPixelInWidth) / widthOfSensor;
  V0 = (numberOfPixelInWidth) / 2;

  double L = 0.2;
  Eigen::Matrix3Xd controlPointsInWorldCoordinate(3, 6);
  Eigen::Matrix3Xd controlPointsInCameraCoordinate(3, 6);

  Eigen::Matrix3d cameraIntrinsicMatrix;
  cameraIntrinsicMatrix << focalLength * mx, 0, V0, 0, focalLength * my, U0, 0,
      0, 1;

  controlPointsInWorldCoordinate.col(0) = Eigen::Vector3d(-L, -L, 0);
  controlPointsInWorldCoordinate.col(1) = Eigen::Vector3d(2 * L, -L, 0.2);
  controlPointsInWorldCoordinate.col(2) = Eigen::Vector3d(L, L, 0.2);
  controlPointsInWorldCoordinate.col(3) = Eigen::Vector3d(-L, L, 0);
  controlPointsInWorldCoordinate.col(4) = Eigen::Vector3d(-2 * L, L, 0);
  controlPointsInWorldCoordinate.col(5) = Eigen::Vector3d(0, 0, 0.5);

  Eigen::Matrix3d rotationMatrix;
  rotationMatrix << 1, 0, 0, 0, 1, 0, 0, 0, 1;

  Eigen::Vector3d translationVector(3, 1);
  translationVector << -0.1, 0.1, 1.2;

  transformPoints(controlPointsInWorldCoordinate, rotationMatrix,
                  translationVector, controlPointsInCameraCoordinate);

  Eigen::Matrix2Xd pointsInCameraPlane(2,
                                       controlPointsInCameraCoordinate.cols());
  projectFromCameraCoordinateToCameraPlane(controlPointsInCameraCoordinate,
                                           cameraIntrinsicMatrix,
                                           pointsInCameraPlane);

  std::vector<cv::Point3d> cvPointsInCameraCoordinate;
  std::vector<cv::Point2d> cvpointsInCameraPlane;
  cvPointsInCameraCoordinate.push_back(cv::Point3d(-L, -L, 0));
  cvPointsInCameraCoordinate.push_back(cv::Point3d(2 * L, -L, 0.2));
  cvPointsInCameraCoordinate.push_back(cv::Point3d(L, L, 0.2));
  cvPointsInCameraCoordinate.push_back(cv::Point3d(-L, L, 0));
  cvPointsInCameraCoordinate.push_back(cv::Point3d(-2 * L, L, 0));
  cvPointsInCameraCoordinate.push_back(cv::Point3d(0, 0, 0.5));

  cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << focalLength * mx, 0, V0, 0,
                          focalLength * my, U0, 0, 0, 1);

  cv::Mat rvec = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat tvec = (cv::Mat_<double>(3, 1) << -0.1, 0.1, 1.2);
  cv::projectPoints(cvPointsInCameraCoordinate, rvec, tvec, cameraMatrix,
                    cv::noArray(), cvpointsInCameraPlane);

  std::cout << "cameraIntrinsicMatrix" << std::endl;
  std::cout << cameraIntrinsicMatrix << std::endl;

  std::cout << cameraMatrix << std::endl;

  Eigen::MatrixXd image(numberOfPixelInHeight, numberOfPixelInWidth);
  image = Eigen::MatrixXd::Zero(numberOfPixelInHeight, numberOfPixelInWidth);
  int U, V;
  for (int i = 0; i < pointsInCameraPlane.cols(); i++) {
    U = int(pointsInCameraPlane(0, i));
    V = int(pointsInCameraPlane(1, i));
    std::cout << U << "," << V << std::endl;
    image(U, V) = 255;
    std::cout << cvpointsInCameraPlane.at(i) << std::endl;
  }
  cv::Mat dst;
  eigen2cv(image, dst);
  std::string fileName = std::string("eigen_camera_projection_result_focal_") +
                         std::to_string(focalLength) + std::string("_.jpg");
  cv::imwrite(fileName, dst);
}

/*
Perspective-n-Point:
Perspective-n-Point is the problem of estimating the pose of a calibrated camera
given a set of n 3D points in the world and their corresponding 2D projections
in the image.
*/

void pnp() {
  double L = 0.2;

  Eigen::Matrix3Xd controlPointsInWorldCoordinate(3, 6);

  controlPointsInWorldCoordinate.col(0) = Eigen::Vector3d(-L, -L, 0);
  controlPointsInWorldCoordinate.col(1) = Eigen::Vector3d(2 * L, -L, 0.2);
  controlPointsInWorldCoordinate.col(2) = Eigen::Vector3d(L, L, 0.2);
  controlPointsInWorldCoordinate.col(3) = Eigen::Vector3d(-L, L, 0);
  controlPointsInWorldCoordinate.col(4) = Eigen::Vector3d(-2 * L, L, 0);
  controlPointsInWorldCoordinate.col(5) = Eigen::Vector3d(0, 0, 0.5);

  std::cout << "Control Points In World Coordinate:" << std::endl;

  std::cout << controlPointsInWorldCoordinate << std::endl;

  Eigen::Matrix3d rotationMatrix;
  Eigen::Vector3d translationVector(3, 1);
  translationVector << -0.1, 0.1, 1.2;

  std::cout << "Translation Vector Ground Truth:" << std::endl;
  std::cout << translationVector << std::endl;

  rotationMatrix << 0.7072945483755065, -0.7061704379962987,
      0.03252282795827704, 0.7061704379962987, 0.7036809008245869,
      -0.07846338199958874, 0.03252282795827704, 0.07846338199958874,
      0.9963863524490802;

  std::cout << "Rotation Matrix Ground Truth:" << std::endl;
  std::cout << rotationMatrix << std::endl;

  Eigen::Matrix3Xd controlPointsInCameraCoordinate(3, 6);
  controlPointsInCameraCoordinate =
      (rotationMatrix * controlPointsInWorldCoordinate).colwise() +
      translationVector;

  /////////////////////////////////////Projecting Into Camera
  /// Plane/////////////////////////////////////

  int numberOfPixelInHeight, numberOfPixelInWidth;
  double heightOfSensor, widthOfSensor;
  double focalLength = 1.50;
  double mx, my, U0, V0;
  numberOfPixelInHeight = 600;
  numberOfPixelInWidth = 800;

  heightOfSensor = 10;
  widthOfSensor = 10;

  // my=-(numberOfPixelInHeight)/heightOfSensor ;
  my = (numberOfPixelInHeight) / heightOfSensor;
  U0 = (numberOfPixelInHeight) / 2;

  mx = (numberOfPixelInWidth) / widthOfSensor;
  V0 = (numberOfPixelInWidth) / 2;

  Eigen::Matrix3d cameraIntrinsicMatrix;

  cameraIntrinsicMatrix << focalLength * mx, 0, V0, 0, focalLength * my, U0, 0,
      0, 1;

  std::cout << "Camera Intrinsic Matrix Ground Truth:" << std::endl;
  std::cout << cameraIntrinsicMatrix << std::endl;

  Eigen::Matrix2Xd pointsInCameraPlane(2,
                                       controlPointsInCameraCoordinate.cols());
  Eigen::Matrix3Xd pointsInCameraPlaneHomogeneous(
      controlPointsInCameraCoordinate.rows(),
      controlPointsInCameraCoordinate.cols());

  // projectFromCameraCoordinateToCameraPlane(controlPointsInCameraCoordinate,cameraIntrinsicMatrix,pointsInCameraPlane,pointsInCameraPlaneHomogeneous);

  std::cout << "Points In Camera Plane:" << std::endl;
  std::cout << pointsInCameraPlane << std::endl;

  std::cout << "=======================================Direct Linear Transform "
               "(DLT)====================================="
            << std::endl;
  // pose_dlt(controlPointsInWorldCoordinate,  pointsInCameraPlane);

  std::cout << "=======================================OpenCV "
               "solvePnPRansac======================================="
            << std::endl;
  // cv::P3P();

  cv::Mat controlPointsInWorldCoordinate_cv, pointsInCameraPlane_cv,
      cameraMatrix, distCoeffs, rvec, tvec;
  eigen2cv(controlPointsInWorldCoordinate, controlPointsInWorldCoordinate_cv);
  eigen2cv(pointsInCameraPlane, pointsInCameraPlane_cv);
  eigen2cv(cameraIntrinsicMatrix, cameraMatrix);

  cv::transpose(controlPointsInWorldCoordinate_cv,
                controlPointsInWorldCoordinate_cv);
  cv::transpose(pointsInCameraPlane_cv, pointsInCameraPlane_cv);

  cv::solvePnPRansac(controlPointsInWorldCoordinate_cv, pointsInCameraPlane_cv,
                     cameraMatrix, distCoeffs, rvec, tvec);
  cv::Mat rotationMatrixFromPnPRansac;
  std::cout << "Rotation Matrix From PnPRansac:" << std::endl;
  cv::Rodrigues(rvec, rotationMatrixFromPnPRansac);
  std::cout << rotationMatrixFromPnPRansac << std::endl;

  std::cout << "Translation Vector From PnPRansac:" << std::endl;
  std::cout << tvec << std::endl;

  std::cout << "=======================================OpenCV "
               "solvePnP======================================="
            << std::endl;
  cv::Mat distCoeffs_, rvec_, tvec_;
  cv::solvePnP(controlPointsInWorldCoordinate_cv, pointsInCameraPlane_cv,
               cameraMatrix, distCoeffs_, rvec_, tvec_);

  cv::Mat rotationMatrixFromsolvePnP;
  cv::Rodrigues(rvec_, rotationMatrixFromsolvePnP);

  std::cout << "Rotation Matrix From PnP:" << std::endl;
  std::cout << rotationMatrixFromsolvePnP << std::endl;

  std::cout << "Translation Vector From PnP:" << std::endl;
  std::cout << tvec_ << std::endl;
}

int main(int argc, char **argv) {
  pnp();
  return 0;
}
