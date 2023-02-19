#include "transformation.hpp"
#include <opencv2/opencv.hpp>

template <typename T> void printArray(std::vector<T> array) {
  for (auto element : array)
    std::cout << element << std::endl;
}

template <typename T> std::vector<T> creatingEllipsoidInWorldCoordinate() {
  std::vector<T> objectPointsInWorldCoordinate;

  float X, Y, Z;

  float phiStepSize, thetaStepSize;
  phiStepSize = 0.7;
  thetaStepSize = 0.6;
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

template <typename T>
cv::Mat matrixFromVectorOfCVPoints(std::vector<T> vertices) {

  // reshape(1)  make Nx3 1-channel matrix out of Nx1 3-channel.
  // t() transpose the Nx3 matrix.

  cv::Mat pointInWorld = cv::Mat(vertices).reshape(1).t();
  std::cout << pointInWorld << std::endl;
  std::cout << "rows: " << pointInWorld.rows << std::endl;
  std::cout << "cols: " << pointInWorld.cols << std::endl;
  std::cout << "channels: " << pointInWorld.channels() << std::endl;
  return pointInWorld;
}

void projectPointcloudInStereoImagePlane() {
  cv::Mat leftCameraRotation_w_c, rightCameraRotation_w_c;
  double rollLeft, pitchLeft, yawLeft, rollRight, pitchRight, yawRight, txLeft,
      tyLeft, tzLeft, txRight, tyRight, tzRight;

  ///////////////// cameras extrinsic /////////////////
  /*





                    Z                        Z
                    ▲                         ▲
                   /                           \
                  /                             \
                 /1 2 3 4     X                  \ 1 2 3 4
    Left Cam   |------------ ⯈                   |------------ ⯈Right cam
              1|                               1 |
              2|                               2 |
              3|                               3 |
             Y |                               Y |
               ⯆                                ⯆



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

  cv::Vec3d thetaLeft, thetaRight;

  rollLeft = 0;
  pitchLeft = +M_PI / 10;
  yawLeft = 0;

  thetaLeft[0] = rollLeft;
  thetaLeft[1] = pitchLeft;
  thetaLeft[2] = yawLeft;

  rollRight = 0;
  pitchRight = -M_PI / 10;
  yawRight = 0;

  thetaRight[0] = rollRight;
  thetaRight[1] = pitchRight;
  thetaRight[2] = yawRight;

  txLeft = -1;
  tyLeft = 0.0;
  tzLeft = +4.0;

  txRight = 1.0;
  tyRight = 0.0;
  tzRight = +4.0;

  leftCameraRotation_w_c = eulerAnglesToRotationMatrix(thetaLeft);
  rightCameraRotation_w_c = eulerAnglesToRotationMatrix(thetaRight);

  cv::Mat leftCameraTranslation_w_c =
      (cv::Mat_<double>(3, 1) << txLeft, tyLeft, tzLeft);
  cv::Mat rightCameraTranslation_w_c =
      (cv::Mat_<double>(3, 1) << txRight, tyRight, tzRight);

  ///////// creating ellipsoid in the world coordinate /////////
  std::vector<cv::Vec3f> objectPointsInWorldCoordinate;

  objectPointsInWorldCoordinate =
      creatingEllipsoidInWorldCoordinate<cv::Vec3f>();

  ///////// camera intrinsic parameters/////////

  int numberOfPixelInHeight, numberOfPixelInWidth;
  double heightOfSensor, widthOfSensor;
  double focalLength = 2.0;
  double mx, my, U0, V0;
  numberOfPixelInHeight = 600;
  numberOfPixelInWidth = 600;

  heightOfSensor = 10;
  widthOfSensor = 10;

  my = (numberOfPixelInHeight) / heightOfSensor;
  U0 = (numberOfPixelInHeight) / 2;

  mx = (numberOfPixelInWidth) / widthOfSensor;
  V0 = (numberOfPixelInWidth) / 2;

  cv::Mat K = (cv::Mat_<double>(3, 3) << focalLength * mx, 0, V0, 0,
               focalLength * my, U0, 0, 0, 1);
  std::vector<cv::Point2f> imagePointsLeftCamera, imagePointsRightCamera;
  cv::projectPoints(objectPointsInWorldCoordinate, leftCameraRotation_w_c.inv(),
                    -leftCameraTranslation_w_c, K, cv::noArray(),
                    imagePointsLeftCamera);

  cv::projectPoints(objectPointsInWorldCoordinate,
                    rightCameraRotation_w_c.inv(), -rightCameraTranslation_w_c,
                    K, cv::noArray(), imagePointsRightCamera);

  ///////// storing images from right and left camera /////////

  std::string fileName;
  cv::Mat rightImage, leftImage;
  int U, V;
  leftImage =
      cv::Mat::zeros(numberOfPixelInHeight, numberOfPixelInWidth, CV_8UC1);

  for (std::size_t i = 0; i < imagePointsLeftCamera.size(); i++) {
    V = int(imagePointsLeftCamera.at(i).x);
    U = int(imagePointsLeftCamera.at(i).y);
    leftImage.at<char>(U, V) = (char)255;
  }

  fileName = std::string("stereoLeftCamera") +
             std::to_string(focalLength) + std::string("_.png");
  cv::imwrite(fileName, leftImage);

  rightImage =
      cv::Mat::zeros(numberOfPixelInHeight, numberOfPixelInWidth, CV_8UC1);
  for (std::size_t i = 0; i < imagePointsRightCamera.size(); i++) {
    V = int(imagePointsRightCamera.at(i).x);
    U = int(imagePointsRightCamera.at(i).y);
    rightImage.at<char>(U, V) = (char)255;
  }

  fileName = std::string("stereoRightCamera") +
             std::to_string(focalLength) + std::string("_.png");
  cv::imwrite(fileName, rightImage);

  ///////// stereoCalibrate /////////

  /**/
  // 1. creating some points in different pose
  std::vector<cv::Vec3f> objectPointsInWorldCoordinate0 =
      creatingEllipsoidInWorldCoordinate<cv::Vec3f>();

  cv::Vec3d theta;
  double roll, pitch, yaw;

  roll = 0;
  pitch = +M_PI / 10;
  yaw = 0;

  theta[0] = roll;
  theta[1] = pitch;
  theta[2] = yaw;

  std::vector<std::vector<cv::Vec3f>> objectPoints;
  objectPoints.push_back(objectPointsInWorldCoordinate0);

  std::vector<std::vector<cv::Point2f>> imagePointsLeft, imagePointsRight;

  imagePointsLeft.push_back(imagePointsLeftCamera);
  imagePointsRight.push_back(imagePointsRightCamera);

  cv::Mat R, T, E, F, perViewErrors;

  int flags = cv::CALIB_FIX_INTRINSIC;
  cv::TermCriteria criteria = cv::TermCriteria(
      cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 1e-6);

  cv::Mat distCoeffs;

  cv::stereoCalibrate(objectPoints, imagePointsLeft, imagePointsRight, K,
                      distCoeffs, K, distCoeffs,
                      cv::Size(numberOfPixelInWidth, numberOfPixelInHeight), R,
                      T, E, F, perViewErrors, flags, criteria);

  std::cout << "R: " << R << "\nT: " << T << "\nE: " << E << "\nF: " << F
            << "\n Per View Errors:" << perViewErrors << std::endl;

  std::cout << rightCameraRotation_w_c.inv() * leftCameraRotation_w_c
            << std::endl;

  std::cout << -leftCameraTranslation_w_c + rightCameraTranslation_w_c
            << std::endl;



  std::cout << R* (-leftCameraTranslation_w_c) + T  << std::endl;

  /*
typedef Vec<float, 3> cv::Vec3f
see Vec

typedef Point3_<float> cv::Point3f

*/
}

int main() { projectPointcloudInStereoImagePlane(); }
