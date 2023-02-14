#include "csv.h"
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

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

std::vector<cv::Point3d>
readPoints(std::string pathToPointFile = "../data/points.csv") {
  io::CSVReader<3> in(pathToPointFile);
  in.read_header(io::ignore_extra_column, "x", "y", "z");
  double x, y, z;
  std::vector<cv::Point3d> objectPoints;

  while (in.read_row(x, y, z)) {
    objectPoints.push_back(cv::Point3d(x, y, z));
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

  for (const auto p : objectPoints)
    std::cout << p << std::endl;

  return objectPoints;
}

void project3DPoint() {

  /*

      OpenCV camera coordinate:

                        Z
                      ▲
                     /
                    /
                   /1 2 3 4     x or u means column
                  |------------ ⯈
                 1|
                 2|
                 3|
                 4|
                  | y or v means row
                  ⯆




      In OpenCV, Point(x=column,y=row). For instance the point in the following
      image can be accessed with

              u
    ------------------------------------------►
    | (0,0) (1,0) (2,0) (3,0) (u,v) (u+1,v)
    | (0,1) (1,1) (2,1) (3,1)
    | (0,2) (1,2) (2,2) (3,2)
  v | (u,v)
    | (u,v+1)
    |
    |
    ▼

  However if you access an image directly, the order is
  mat.at<type>(row,column). So the following will return the same value:
  mat.at<type>(row,column)
  mat.at<type>(cv::Point(column,row))

  X
  --------column---------►
  | mat.at<type>(0,0) mat.at<type>(0,1) mat.at<type>(0,2) mat.at<type>(0,3)
  | mat.at<type>(1,0) mat.at<type>(1,1) mat.at<type>(1,2) mat.at<type>(1,3)
  | mat.at<type>(2,0) mat.at<type>(2,1) mat.at<type>(2,2) mat.at<type>(2,3)
  y |
  row
  |
  |
  ▼



      The parameters fx=f*mx  and fy=f*my  where mx=1/width and my=1/height
      meaning size of 1 pixel in x and y

      mx=1/width
      my=1/height

      cx=Width/2;
      cy=Height/2 ;

      fx=f*mx

      k=[fx  0  cx
         0  fy  cy
         0  0   1 ]

                      Z
                      ▲
                     /
                    /
                   /1 2 3 4     X
                  |------------ ⯈
                 1|
                 2|
                 3|
                 4|
                  | Y
                  ⯆


      mat.at<type>(row,column)
      mat.at<type>(cv::Point(column,row))

      */

  ///////////////// camera intrinsic /////////////////
  int numberOfPixelInHeight, numberOfPixelInWidth;
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

  cv::Mat distortionCoefficient = (cv::Mat_<double>(5, 1) << 0, 0, 0, 0, 0);
  std::cout << "camera distortion coefficient:\n"
            << distortionCoefficient << std::endl;

  ///////////////// cameras extrinsic /////////////////
  /*

    P[x,y,z,w]

                                           Z
                                            ▲
                                             \
                                              \
                                               \ 1 2 3 4
                                               |------------  camera
                                             1 |
                                             2 |
                                             3 |
                                             Y |
                                               ⯆


                  Z
                  ▲
                 /
                /
               /1 2 3 4     X
  world      |------------ ⯈
            1|
            2|
            3|
           Y |
             ⯆
  */

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

  std::vector<cv::Point3d> objectPointsInWorldesCoordinate =
      readPoints("../data/points.csv");

  ///////////////// 3D points from world /////////////////

  std::vector<cv::Point2d> projectedPointsInCamera;

  cv::Mat R_c_w = R_w_c.inv();
  cv::Mat T_c_w = -T_w_c;

  /*
   `rvec` and `tvec` are the rotation and translation that transform object pose
from world coordinate into camera's coordinate, namely R_c_w=R_w_c.inv() and
T_c_w=-T_w_c



  */

  std::cout << "======= projecting 3D points into camera  unsing "
               "OpenCV======="
            << std::endl;

  cv::projectPoints(objectPointsInWorldesCoordinate, R_c_w, T_c_w, cameraMatrix,
                    distortionCoefficient, projectedPointsInCamera);

  std::cout << "projected point in camera" << std::endl;
  for (const auto p : projectedPointsInCamera)
    std::cout << "row: " << p.y << ","
              << " column: " << p.x << std::endl;

  std::cout
      << "======= projecting 3D points into camera unsing P=K[R|t] ======="
      << std::endl;

  //   1)P=K[R|t]
  cv::Mat P(3, 4, cv::DataType<double>::type),
      R_t(3, 4, cv::DataType<double>::type);
  R_t.at<double>(0, 0) = R_c_w.at<double>(0, 0);
  R_t.at<double>(0, 1) = R_c_w.at<double>(0, 1);
  R_t.at<double>(0, 2) = R_c_w.at<double>(0, 2);

  R_t.at<double>(1, 0) = R_c_w.at<double>(1, 0);
  R_t.at<double>(1, 1) = R_c_w.at<double>(1, 1);
  R_t.at<double>(1, 2) = R_c_w.at<double>(1, 2);

  R_t.at<double>(2, 0) = R_c_w.at<double>(2, 0);
  R_t.at<double>(2, 1) = R_c_w.at<double>(2, 1);
  R_t.at<double>(2, 2) = R_c_w.at<double>(2, 2);

  R_t.at<double>(0, 3) = T_c_w.at<double>(0, 0);
  R_t.at<double>(1, 3) = T_c_w.at<double>(1, 0);
  R_t.at<double>(2, 3) = T_c_w.at<double>(2, 0);

  P = cameraMatrix * R_t;
  std::cout << P << std::endl;

  cv::Mat1d pointInWorldCoordinateHomogeneous(4, 1);

  cv::Mat pHomogeneous(3, 1, cv::DataType<double>::type);

  std::cout << "projected point in camera" << std::endl;

  for (auto const &point : objectPointsInWorldesCoordinate) {
    pointInWorldCoordinateHomogeneous.at<double>(0, 0) = point.x;
    pointInWorldCoordinateHomogeneous.at<double>(1, 0) = point.y;
    pointInWorldCoordinateHomogeneous.at<double>(2, 0) = point.z;
    pointInWorldCoordinateHomogeneous.at<double>(3, 0) = 1;

    pHomogeneous = P * pointInWorldCoordinateHomogeneous;

    std::cout << "row: "
              << pHomogeneous.at<double>(1, 0) / pHomogeneous.at<double>(2, 0)
              << " , column: "
              << pHomogeneous.at<double>(0, 0) / pHomogeneous.at<double>(2, 0)
              << std::endl;
  }

  std::cout
      << "======= projecting 3D points into camera unsing P=KR[I-C] ======="
      << std::endl;
  /*
    cv::Mat I_C = cv::Mat::zeros(3, 4, cv::DataType<double>::type);

    I_C.at<double>(0, 0) = 1;
    I_C.at<double>(1, 1) = 1;
    I_C.at<double>(2, 2) = 1;

    I_C.at<double>(0, 3) = -T_w_c.at<double>(0, 0);
    I_C.at<double>(1, 3) = -T_w_c.at<double>(1, 0);
    I_C.at<double>(2, 3) = -T_w_c.at<double>(2, 0);

    P = cameraMatrix * R_c_w * I_C;
    std::cout << P << std::endl;

    std::cout << "projected point in camera" << std::endl;

    for (auto const &point : objectPointsInWorldesCoordinate) {
      pointInWorldCoordinateHomogeneous.at<double>(0, 0) = point.x;
      pointInWorldCoordinateHomogeneous.at<double>(1, 0) = point.y;
      pointInWorldCoordinateHomogeneous.at<double>(2, 0) = point.z;
      pointInWorldCoordinateHomogeneous.at<double>(3, 0) = 1;

      pHomogeneous = P * pointInWorldCoordinateHomogeneous;

      std::cout << "row: "
                << pHomogeneous.at<double>(1, 0) / pHomogeneous.at<double>(2, 0)
                << " , column: "
                << pHomogeneous.at<double>(0, 0) / pHomogeneous.at<double>(2, 0)
                << std::endl;
    }
  */

  std::cout << "======= decomposing Projection Matrix ======= " << std::endl;

  std::cout << "=======Ground Truth=======" << std::endl;

  std::cout << "Rotation Matrix (Ground Truth)" << std::endl;
  std::cout << R_c_w << std::endl;

  std::cout << "Translation Matrix (Ground Truth)" << std::endl;
  std::cout << T_c_w << std::endl;

  std::cout << "Camera Matrix (Ground Truth)" << std::endl;
  std::cout << cameraMatrix << std::endl;

  cv::Mat calculatedCameraMatrix, calculatedRotation, calculatedTranslation;

  cv::decomposeProjectionMatrix(P, calculatedCameraMatrix, calculatedRotation,
                                calculatedTranslation);

  std::cout << "Computed Rotation Matrix (OpenCV)" << std::endl;
  std::cout << calculatedRotation << std::endl;

  std::cout << "Computed Translation Matrix (OpenCV)" << std::endl;
  std::cout << calculatedTranslation / calculatedTranslation.at<double>(3, 0)
            << std::endl;

  std::cout<<"Computed Translation Matrix (OpenCV)" <<std::endl;
    //std::cout<<calculatedTranslation/calculatedTranslation.at<double>(3,0) <<std::endl;
    cv::Mat tempT=(cv::Mat_<double>(3,1)<<
                   calculatedTranslation.at<double>(0,0)/calculatedTranslation.at<double>(3,0),
                   calculatedTranslation.at<double>(1,0)/calculatedTranslation.at<double>(3,0),
                   calculatedTranslation.at<double>(2,0)/calculatedTranslation.at<double>(3,0));
    std::cout<<-calculatedRotation*tempT<<std::endl;

    std::cout << "Computed Camera Matrix (OpenCV)" << std::endl;
    std::cout << calculatedCameraMatrix << std::endl;

  // https://ros-developer.com/2019/01/01/decomposing-projection-using-opencv-and-c/

  //  cv::Mat R_t;
  //  cv::hconcat(R_c_w, T_w_c, R_t);
  //  cv::Mat tionMatrix = cameraMatrix * R_t;

  //  std::cout << cameraMatrix * P << std::endl;

  //  std::cout << projectionMatrix << std::endl;

  // cv::convertPointsToHomogeneous(objectPointsInWorldCoordinate,
  //                              objectPointsInWorldCoordinateHomogeneous);

  // cv::Point3d pHomogeneous=cameraMatrix
  // *P*pointInWorldCoordinateHomogeneous ;
}

int main(int argc, char **argv) { project3DPoint(); }
