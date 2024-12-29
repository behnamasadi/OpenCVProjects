#include "collection_adapters.hpp"
#include "csv.h"
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <rerun.hpp>
#include <rerun/archetypes/arrows3d.hpp>

// https://www.cnblogs.com/shengguang/p/5932522.html
void HouseHolderQR(const cv::Mat &A, cv::Mat &Q, cv::Mat &R) {
  assert(A.channels() == 1);
  assert(A.rows >= A.cols);
  auto sign = [](double value) { return value >= 0 ? 1 : -1; };
  const auto totalRows = A.rows;
  const auto totalCols = A.cols;
  R = A.clone();
  Q = cv::Mat::eye(totalRows, totalRows, A.type());
  for (int col = 0; col < A.cols; ++col) {
    cv::Mat matAROI =
        cv::Mat(R, cv::Range(col, totalRows), cv::Range(col, totalCols));
    cv::Mat y = matAROI.col(0);
    auto yNorm = norm(y);
    cv::Mat e1 = cv::Mat::eye(y.rows, 1, A.type());
    cv::Mat w = y + sign(y.at<double>(0, 0)) * yNorm * e1;
    cv::Mat v = w / norm(w);
    cv::Mat vT;
    cv::transpose(v, vT);
    cv::Mat I = cv::Mat::eye(matAROI.rows, matAROI.rows, A.type());
    cv::Mat I_2VVT = I - 2 * v * vT;
    cv::Mat matH = cv::Mat::eye(totalRows, totalRows, A.type());
    cv::Mat matHROI =
        cv::Mat(matH, cv::Range(col, totalRows), cv::Range(col, totalRows));
    I_2VVT.copyTo(matHROI);
    R = matH * R;
    Q = Q * matH;
  }
}

cv::Mat createImage(double focalLength, int numberOfPixelInHeight,
                    int numberOfPixelInWidth,
                    std::vector<cv::Point2d> projectedPoints,
                    std::string fileName) {

  double row, col;

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
    // std::cout<<row <<"," <<col  <<std::endl;
    cameraImage.at<char>(int(row), int(col)) = char(255);
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

  cv::Mat distortionCoefficient = (cv::Mat_<double>(5, 1) << 0, 0, 0, 0, 0);
  std::cout << "camera distortion coefficient:\n"
            << distortionCoefficient << std::endl;

  ///////////////// cameras extrinsic /////////////////
  /*

    P[x,y,z,w]




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
      readPoints("../../data/points.csv");

  const auto rec = rerun::RecordingStream("camera_projection_matrix");
  rec.spawn().exit_on_failure();
  // OpenCV X=Right, Y=Down, Z=Forward
  rec.log_static("world", rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN);
  std::vector<rerun::components::Position3D> point3d_positions;
  std::vector<float> point_sizes; // Define a vector for point sizes

  // Log the arrows to the Rerun Viewer
  rec.log("world/xyz",
          rerun::Arrows3D::from_vectors(
              {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}})
              .with_colors({{255, 0, 0}, {0, 255, 0}, {0, 0, 255}}));

  // objectPointsInCameraCoordinate;
  float x, y, z;

  for (std::size_t i = 0; i < objectPointsInWorldesCoordinate.size(); i++) {
    x = objectPointsInWorldesCoordinate[i].x;
    y = objectPointsInWorldesCoordinate[i].y;
    z = objectPointsInWorldesCoordinate[i].z;
    point3d_positions.push_back({x, y, z});
    point_sizes.push_back(0.1);
  }

  rec.log("world/points",
          rerun::Points3D(point3d_positions).with_radii(point_sizes));

  // Extract translation vector
  rerun::Vec3D translation(T_w_c.at<double>(0, 0), T_w_c.at<double>(1, 0),
                           T_w_c.at<double>(2, 0));

  // Extract rotation matrix as std::array<float, 9> in
  // column-major order
  std::array<float, 9> rotation_data = {
      static_cast<float>(R_w_c.at<double>(0, 0)),
      static_cast<float>(R_w_c.at<double>(1, 0)),
      static_cast<float>(R_w_c.at<double>(2, 0)),

      static_cast<float>(R_w_c.at<double>(0, 1)),
      static_cast<float>(R_w_c.at<double>(1, 1)),
      static_cast<float>(R_w_c.at<double>(2, 1)),

      static_cast<float>(R_w_c.at<double>(0, 2)),
      static_cast<float>(R_w_c.at<double>(1, 2)),
      static_cast<float>(R_w_c.at<double>(2, 2)),
  };

  rerun::Mat3x3 rotation_matrix;
  rotation_matrix = rotation_data;

  // Log the data
  std::string camera_name = "world/camera";

  rec.log(camera_name,
          rerun::Pinhole::from_focal_length_and_resolution(
              {float(focalLength * mx), float(focalLength * my)},
              {float(numberOfPixelInWidth), float(numberOfPixelInHeight)}));

  rec.log(camera_name, rerun::Transform3D(translation, rotation_matrix));

  ///////////////// 3D points from world /////////////////

  std::vector<cv::Point2d> projectedPointsInCamera;

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

  // Log the image to the camera entity in the hierarchy
  rec.log("world/camera/image/rgb",
          rerun::Image::from_greyscale8(
              img, {numberOfPixelInWidth, numberOfPixelInHeight}));

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
  std::cout << "projection matrix:" << P << std::endl;

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

  std::cout << "========projecting 3D points from camera coordinate ========"
            << std::endl;

  std::vector<cv::Point3d> objectPointsInCameraCoordinate;

  cv::Mat Transformation_c_w = cv::Mat::zeros(4, 4, cv::DataType<double>::type);

  Transformation_c_w.at<double>(0, 0) = R_c_w.at<double>(0, 0);
  Transformation_c_w.at<double>(0, 1) = R_c_w.at<double>(0, 1);
  Transformation_c_w.at<double>(0, 2) = R_c_w.at<double>(0, 2);

  Transformation_c_w.at<double>(1, 0) = R_c_w.at<double>(1, 0);
  Transformation_c_w.at<double>(1, 1) = R_c_w.at<double>(1, 1);
  Transformation_c_w.at<double>(1, 2) = R_c_w.at<double>(1, 2);

  Transformation_c_w.at<double>(2, 0) = R_c_w.at<double>(2, 0);
  Transformation_c_w.at<double>(2, 1) = R_c_w.at<double>(2, 1);
  Transformation_c_w.at<double>(2, 2) = R_c_w.at<double>(2, 2);

  Transformation_c_w.at<double>(0, 3) = T_c_w.at<double>(0, 0);
  Transformation_c_w.at<double>(1, 3) = T_c_w.at<double>(1, 0);
  Transformation_c_w.at<double>(2, 3) = T_c_w.at<double>(2, 0);
  Transformation_c_w.at<double>(3, 3) = 1;

  for (auto const &p : objectPointsInWorldesCoordinate) {

    pointInWorldCoordinateHomogeneous.at<double>(0, 0) = p.x;
    pointInWorldCoordinateHomogeneous.at<double>(1, 0) = p.y;
    pointInWorldCoordinateHomogeneous.at<double>(2, 0) = p.z;
    pointInWorldCoordinateHomogeneous.at<double>(3, 0) = 1;

    cv::Mat1d pointInCameraCoordinateHomogeneous(4, 1);
    pointInCameraCoordinateHomogeneous =
        Transformation_c_w * pointInWorldCoordinateHomogeneous;
    //    cv::convertPointsFromHomogeneous(Transformation_c_w *
    //                                         pointInWorldCoordinateHomogeneous,
    //                                     pointInCameraCoordinate);
    cv::Point3d tmp_p;
    tmp_p.x = pointInCameraCoordinateHomogeneous.at<double>(0, 0) /
              pointInCameraCoordinateHomogeneous.at<double>(3, 0);
    tmp_p.y = pointInCameraCoordinateHomogeneous.at<double>(1, 0) /
              pointInCameraCoordinateHomogeneous.at<double>(3, 0);
    tmp_p.z = pointInCameraCoordinateHomogeneous.at<double>(2, 0) /
              pointInCameraCoordinateHomogeneous.at<double>(3, 0);
    objectPointsInCameraCoordinate.push_back(tmp_p);
  }

  cv::Mat projectedPointsMatrixHomogeneous =
      cameraMatrix * cv::Mat(objectPointsInCameraCoordinate).reshape(1).t();

  for (int i = 0; i < projectedPointsMatrixHomogeneous.cols; i++) {
    projectedPointsMatrixHomogeneous.at<double>(0, i) /=
        projectedPointsMatrixHomogeneous.at<double>(2, i);
    projectedPointsMatrixHomogeneous.at<double>(1, i) /=
        projectedPointsMatrixHomogeneous.at<double>(2, i);
    projectedPointsMatrixHomogeneous.at<double>(2, i) /=
        projectedPointsMatrixHomogeneous.at<double>(2, i);
  }
  std::cout << projectedPointsMatrixHomogeneous << std::endl;

  std::cout << "=======Ground Truth=======" << std::endl;

  std::cout << "Rotation Matrix R_c_w (Ground Truth)" << std::endl;
  std::cout << R_c_w << std::endl;

  std::cout << "Translation Matrix T_c_w (Ground Truth)" << std::endl;
  std::cout << T_c_w << std::endl;

  std::cout << "Camera Matrix (Ground Truth)" << std::endl;
  std::cout << cameraMatrix << std::endl;

  cv::Mat calculatedCameraMatrix, calculatedRotation_c_w, C_homogeneous;

  cv::decomposeProjectionMatrix(P, calculatedCameraMatrix,
                                calculatedRotation_c_w, C_homogeneous);

  std::cout << "Calculated Rotation Matrix (OpenCV)" << std::endl;
  std::cout << calculatedRotation_c_w << std::endl;

  std::cout << "Calculated Translation Matrix (OpenCV)" << std::endl;
  cv::Mat C = (cv::Mat_<double>(3, 1) << C_homogeneous.at<double>(0, 0) /
                                             C_homogeneous.at<double>(3, 0),
               C_homogeneous.at<double>(1, 0) / C_homogeneous.at<double>(3, 0),
               C_homogeneous.at<double>(2, 0) / C_homogeneous.at<double>(3, 0));
  std::cout << -calculatedRotation_c_w * C << std::endl;

  std::cout << "Computed Camera Matrix (OpenCV)" << std::endl;
  std::cout << calculatedCameraMatrix << std::endl;

  /*

      P=KR[I|-X0]=[H_inf3x3|h3x1]
      KR=H_inf3x3
      1)X0
      -KRX0=h3x1 => X0=-(KR)^-1*h3x1 ==>X0=-(H_inf3x3)^-1*h3x1
      2)K,R

      KR=H_inf3x3 =>(KR)^-1= H_inf3x3^-1 =>R^-1*K^-1=H_inf3x3^-1 | R^-1*K^-1=Q*R
     => R=Q^-1, K=R^-1 H_inf3x3^-1=Q*R       |

  */

  cv::Mat H_inf3x3 =
      (cv::Mat_<double>(3, 3) << P.at<double>(0, 0), P.at<double>(0, 1),
       P.at<double>(0, 2), P.at<double>(1, 0), P.at<double>(1, 1),
       P.at<double>(1, 2), P.at<double>(2, 0), P.at<double>(2, 1),
       P.at<double>(2, 2));

  cv::Mat h3x1 = (cv::Mat_<double>(3, 1) << P.at<double>(0, 3),
                  P.at<double>(1, 3), P.at<double>(2, 3));

  cv::Mat Q, R;
  cv::Mat H_inf3x3_inv = H_inf3x3.inv();
  // R=Q^-1, K=R^-1

  HouseHolderQR(H_inf3x3_inv, Q, R);
  cv::Mat K = R.inv();

  std::cout << "========Decomposing Using My Code======" << std::endl;
  // due to homogeneity we divide it by last element
  std::cout << "Estimated Camera Matrix\n"
            << K / K.at<double>(2, 2) << std::endl;

  cv::Mat rotationMatrix = Q.inv();
  std::cout << "Estimated Camera Rotation\n"
            << rotationMatrix * -1 << std::endl;

  std::cout << "Estimated Camera Translation" << std::endl;

  // t=-R*C, Q.inv()=R
  std::cout << -1 * (-Q.inv() * (-H_inf3x3.inv() * h3x1)) << std::endl;

  std::cout << "========3D World Unit Vector======" << std::endl;

  R_c_w = cv::Mat::eye(3, 3, cv::DataType<double>::type);
  T_c_w = cv::Mat::zeros(3, 1, cv::DataType<double>::type);

  std::cout << R_c_w << std::endl;
  std::cout << T_c_w << std::endl;

  cv::projectPoints(objectPointsInWorldesCoordinate, R_c_w, T_c_w, cameraMatrix,
                    distortionCoefficient, projectedPointsInCamera);

  cv::Mat projectedPointsHomogenous;
  int cols = projectedPointsInCamera.size();
  int rows = 3;

  projectedPointsHomogenous.create(rows, cols, CV_64FC1);
  for (int j = 0; j < cols; j++) {
    projectedPointsHomogenous.at<double>(0, j) = projectedPointsInCamera[j].x;
    projectedPointsHomogenous.at<double>(1, j) = projectedPointsInCamera[j].y;
    projectedPointsHomogenous.at<double>(2, j) = 1;
  }

  /*
  The inverse of a 3x3 matrix:
  | a11 a12 a13 |-1
  | a21 a22 a23 |    =  1/DET * A^-1
  | a31 a32 a33 |

  with A^-1  =

  |  a33a22-a32a23  -(a33a12-a32a13)   a23a12-a22a13 |
  |-(a33a21-a31a23)   a33a11-a31a13  -(a23a11-a21a13)|
  |  a32a21-a31a22  -(a32a11-a31a12)   a22a11-a21a12 |

  and DET  =  a11(a33a22-a32a23) - a21(a33a12-a32a13) + a31(a23a12-a22a13)

  Camera Matrix:

  |fx 0 cx|
  |0 fy cy|
  |0 0  1 |

  Rays are  A^-1*p:

   1    |fy 0   -fycx|  |u|
  ----- |0  fx -cy*fx| *|v| = [ (u- cx)/fx, (v-cx)/fy, 1]
  fx*fy |0  0   fy*fx|  |1|

  */

  cv::Mat rays = cameraMatrix.inv() *
                 projectedPointsHomogenous; // put in world coordinates
  std::cout << "camera rays" << std::endl;
  std::cout << rays << std::endl;

  std::cout << "unit vector (normalized camera rays)" << std::endl;
  rays *= 1 / cv::norm(rays);
  std::cout << rays << std::endl;
}

int main(int argc, char **argv) { project3DPoint(); }
