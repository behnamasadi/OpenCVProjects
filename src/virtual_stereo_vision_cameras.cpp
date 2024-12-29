#include "collection_adapters.hpp"
#include "transformation.hpp"
#include <opencv2/opencv.hpp>
#include <rerun.hpp>
#include <rerun/archetypes/arrows3d.hpp>

cv::Mat createImage(double focalLength, int numberOfPixelInHeight,
                    int numberOfPixelInWidth,
                    std::vector<cv::Point2f> projectedPoints,
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
    // std::cout << row << "," << col << std::endl;
    if (int(row) < numberOfPixelInHeight && int(col) < numberOfPixelInWidth)
      cameraImage.at<char>(int(row), int(col)) = char(255);
    else {
      std::cout << row << "," << col << "is out of image" << std::endl;
    }
  }

  cv::imwrite(fileName, cameraImage);
  return cameraImage;
}

template <typename T> rerun::Vec3D getRerunTranslationFromCvMat(cv::Mat t) {

  // Extract translation vector
  rerun::Vec3D translation(t.at<T>(0, 0), t.at<T>(1, 0), t.at<T>(2, 0));

  return translation;
}

template <typename T> rerun::Mat3x3 getRerunRotationFromCvMat(cv::Mat R) {

  // Extract rotation matrix as std::array<float, 9> in
  // column-major order
  std::array<float, 9> rotation_data = {
      static_cast<float>(R.at<double>(0, 0)),
      static_cast<float>(R.at<double>(1, 0)),
      static_cast<float>(R.at<double>(2, 0)),

      static_cast<float>(R.at<double>(0, 1)),
      static_cast<float>(R.at<double>(1, 1)),
      static_cast<float>(R.at<double>(2, 1)),

      static_cast<float>(R.at<double>(0, 2)),
      static_cast<float>(R.at<double>(1, 2)),
      static_cast<float>(R.at<double>(2, 2)),
  };

  return rotation_data;
}

template <typename T> void printArray(std::vector<T> array) {
  for (auto element : array)
    std::cout << element << std::endl;
}

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

template <typename T>
std::vector<T> createChessboardCornerGrid(int rows = 7, int cols = 9,
                                          float squareSize = 0.5) {
  std::vector<T> objectPointsInWorldCoordinate;

  for (int row = -rows / 2; row < rows / 2; ++row) {
    for (int col = -cols / 2; col < cols / 2; ++col) {
      float X = col * squareSize;
      float Y = row * squareSize;
      float Z = 0.0f; // Chessboard lies on the Z=0 plane

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


  */

  cv::Vec3d thetaLeft, thetaRight;

  rollLeft = 0;
  // pitchLeft = +M_PI / 60;
  pitchLeft = 0;
  yawLeft = 0;

  thetaLeft[0] = rollLeft;
  thetaLeft[1] = pitchLeft;
  thetaLeft[2] = yawLeft;

  rollRight = 0;
  // pitchRight = -M_PI / 60;
  pitchRight = 0;
  yawRight = 0;

  thetaRight[0] = rollRight;
  thetaRight[1] = pitchRight;
  thetaRight[2] = yawRight;

  txLeft = -1;
  tyLeft = 0.0;
  tzLeft = -4.0;

  txRight = 1.0;
  tyRight = 0.0;
  tzRight = -4.0;

  leftCameraRotation_w_c = eulerAnglesToRotationMatrix(thetaLeft);
  rightCameraRotation_w_c = eulerAnglesToRotationMatrix(thetaRight);

  cv::Mat leftCameraTranslation_w_c =
      (cv::Mat_<double>(3, 1) << txLeft, tyLeft, tzLeft);
  cv::Mat rightCameraTranslation_w_c =
      (cv::Mat_<double>(3, 1) << txRight, tyRight, tzRight);

  ///////// creating ellipsoid in the world coordinate /////////
  std::vector<cv::Vec3f> objectPointsInWorldCoordinate;

  objectPointsInWorldCoordinate = createEllipsoidInWorldCoordinate<cv::Vec3f>();

  // objectPointsInWorldCoordinate = createChessboardCornerGrid<cv::Vec3f>();

  ///////// camera intrinsic parameters/////////

  unsigned int numberOfPixelInHeight, numberOfPixelInWidth;
  double heightOfSensor, widthOfSensor;
  double focalLength = 4.0;
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

  cv::Mat leftCameraRotation_c_w =
      leftCameraRotation_w_c.t(); // Transpose (or inverse) of R_w_c
  cv::Mat leftCamera_T_c_w =
      -leftCameraRotation_c_w *
      leftCameraTranslation_w_c; // Correct transformation of the translation
                                 // vector

  cv::projectPoints(objectPointsInWorldCoordinate, leftCameraRotation_c_w,
                    leftCamera_T_c_w, K, cv::noArray(), imagePointsLeftCamera);

  cv::Mat rightCameraRotation_c_w =
      rightCameraRotation_w_c.t(); // Transpose (or inverse) of R_w_c
  cv::Mat rightCamera_T_c_w =
      -rightCameraRotation_c_w *
      rightCameraTranslation_w_c; // Correct transformation of the translation
                                  // vector

  cv::projectPoints(objectPointsInWorldCoordinate, rightCameraRotation_c_w,
                    rightCamera_T_c_w, K, cv::noArray(),
                    imagePointsRightCamera);

  const auto rec = rerun::RecordingStream("stereo_vision");
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

  for (std::size_t i = 0; i < objectPointsInWorldCoordinate.size(); i++) {
    x = objectPointsInWorldCoordinate[i][0];
    y = objectPointsInWorldCoordinate[i][1];
    z = objectPointsInWorldCoordinate[i][2];
    point3d_positions.push_back({x, y, z});
    point_sizes.push_back(0.05);
  }

  rec.log("world/points",
          rerun::Points3D(point3d_positions).with_radii(point_sizes));

  // Extract translation vector
  rerun::Vec3D translation_left =
      getRerunTranslationFromCvMat<double>(leftCameraTranslation_w_c);
  rerun::Vec3D translation_right =
      getRerunTranslationFromCvMat<double>(rightCameraTranslation_w_c);

  rerun::Mat3x3 rotation_matrix_left =
      getRerunRotationFromCvMat<double>(leftCameraRotation_w_c);
  rerun::Mat3x3 rotation_matrix_right =
      getRerunRotationFromCvMat<double>(rightCameraRotation_w_c);

  // Log the data
  std::string camera_name_left = "world/camera_left";
  std::string camera_name_right = "world/camera_right";

  rec.log(camera_name_left,
          rerun::Pinhole::from_focal_length_and_resolution(
              {float(focalLength * mx), float(focalLength * my)},
              {float(numberOfPixelInWidth), float(numberOfPixelInHeight)}));

  rec.log(camera_name_right,
          rerun::Pinhole::from_focal_length_and_resolution(
              {float(focalLength * mx), float(focalLength * my)},
              {float(numberOfPixelInWidth), float(numberOfPixelInHeight)}));

  rec.log(camera_name_left,
          rerun::Transform3D(translation_left, rotation_matrix_left));

  rec.log(camera_name_right,
          rerun::Transform3D(translation_right, rotation_matrix_right));

  ////////////////////////////////////////////////////////////////////////

  std::string fileName;
  fileName = std::string("image_left") + std::to_string(focalLength) +
             std::string("_.png");

  cv::Mat img_left =
      createImage(focalLength, numberOfPixelInHeight, numberOfPixelInWidth,
                  imagePointsLeftCamera, fileName);

  // Log the image to the camera entity in the hierarchy
  rec.log("world/camera_left/image/rgb",
          rerun::Image::from_greyscale8(
              img_left, {numberOfPixelInWidth, numberOfPixelInHeight}));

  fileName = std::string("image_right") + std::to_string(focalLength) +
             std::string("_.png");

  cv::Mat img_right =
      createImage(focalLength, numberOfPixelInHeight, numberOfPixelInWidth,
                  imagePointsRightCamera, fileName);

  // Log the image to the camera entity in the hierarchy
  rec.log("world/camera_right/image/rgb",
          rerun::Image::from_greyscale8(
              img_right, {numberOfPixelInWidth, numberOfPixelInHeight}));

  ///////// stereoCalibrate /////////

  std::vector<std::vector<cv::Vec3f>> objectPoints;
  objectPoints.push_back(objectPointsInWorldCoordinate);

  std::vector<std::vector<cv::Point2f>> imagePointsLeft, imagePointsRight;

  imagePointsLeft.push_back(imagePointsLeftCamera);
  imagePointsRight.push_back(imagePointsRightCamera);

  cv::Mat R, T, E, F, perViewErrors;

  int flags = cv::CALIB_FIX_INTRINSIC;
  cv::TermCriteria criteria = cv::TermCriteria(
      cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 1e-6);

  cv::Mat distortionCoefficient = (cv::Mat_<double>(5, 1) << 0, 0, 0, 0, 0);

  double rms =
      cv::stereoCalibrate(objectPoints, imagePointsLeft, imagePointsRight, K,
                          distortionCoefficient, K, distortionCoefficient,
                          cv::Size(numberOfPixelInWidth, numberOfPixelInHeight),
                          R, T, E, F, perViewErrors, flags, criteria);

  std::cout << "rms: " << rms << std::endl;

  std::cout << "R: " << R << "\nT: " << T << "\nE: " << E << "\nF: " << F
            << "\n Per View Errors:" << perViewErrors << std::endl;

  std::cout << R * (-leftCameraTranslation_w_c) + T << std::endl;

  cv::Mat rightCameraCenter = -R.t() * T; // Compute the right camera center
  std::cout << "Right camera center in left camera frame:\n"
            << rightCameraCenter << std::endl;

  // Rectification matrices
  cv::Mat R1, R2, P1, P2, Q;
  cv::stereoRectify(K, distortionCoefficient, K, distortionCoefficient,
                    cv::Size(numberOfPixelInWidth, numberOfPixelInHeight), R, T,
                    R1, R2, P1, P2, Q);

  // Compute rectification maps
  cv::Mat map1x, map1y, map2x, map2y;
  cv::initUndistortRectifyMap(
      K, distortionCoefficient, R1, P1,
      cv::Size(numberOfPixelInWidth, numberOfPixelInHeight), CV_32FC1, map1x,
      map1y);
  cv::initUndistortRectifyMap(
      K, distortionCoefficient, R2, P2,
      cv::Size(numberOfPixelInWidth, numberOfPixelInHeight), CV_32FC1, map2x,
      map2y);

  // Rectify the images
  cv::Mat rectifiedImageLeft, rectifiedImageRight;
  cv::remap(img_left, rectifiedImageLeft, map1x, map1y, cv::INTER_LINEAR);
  cv::remap(img_right, rectifiedImageRight, map2x, map2y, cv::INTER_LINEAR);

  // Save or display the rectified images
  cv::imwrite("rectified_image_left.png", rectifiedImageLeft);
  cv::imwrite("rectified_image_right.png", rectifiedImageRight);

  // Log rectified images to Rerun
  rec.log(
      "world/camera_left/image/rectified",
      rerun::Image::from_greyscale8(
          rectifiedImageLeft, {numberOfPixelInWidth, numberOfPixelInHeight}));

  rec.log(
      "world/camera_right/image/rectified",
      rerun::Image::from_greyscale8(
          rectifiedImageRight, {numberOfPixelInWidth, numberOfPixelInHeight}));

  // Triangulate points
  cv::Mat points4D;
  cv::triangulatePoints(P1, P2, imagePointsLeftCamera, imagePointsRightCamera,
                        points4D);

  // Convert points from homogeneous to 3D (divide by w)
  std::vector<cv::Point3f> triangulatedPoints;
  for (int i = 0; i < points4D.cols; ++i) {
    cv::Point3f point;
    point.x = points4D.at<float>(0, i) / points4D.at<float>(3, i);
    point.y = points4D.at<float>(1, i) / points4D.at<float>(3, i);
    point.z = points4D.at<float>(2, i) / points4D.at<float>(3, i);
    triangulatedPoints.push_back(point);
  }

  //////////////////////////////////////////////////////////////////////
  cv::Mat R_c_w =
      leftCameraRotation_w_c.t(); // Rotation from world to left camera
  cv::Mat T_c_w =
      -R_c_w *
      leftCameraTranslation_w_c; // Translation from world to left camera

  for (auto &point : triangulatedPoints) {
    cv::Mat pointInCamera =
        (cv::Mat_<double>(3, 1) << point.x, point.y, point.z);
    cv::Mat pointInWorld = R_c_w * pointInCamera + T_c_w;

    point.x = pointInWorld.at<double>(0);
    point.y = pointInWorld.at<double>(1);
    point.z = pointInWorld.at<double>(2);
  }
  //////////////////////////////////////////////////////////////////////

  // Log the triangulated points to Rerun
  std::vector<rerun::components::Position3D> triangulated3d_positions;
  for (const auto &point : triangulatedPoints) {
    triangulated3d_positions.push_back({point.x, point.y, point.z});
  }
  rec.log("world/triangulated_points",
          rerun::Points3D(triangulated3d_positions)
              .with_radii(std::vector<float>(triangulatedPoints.size(), 0.05)));

  // Optional: Print or save the triangulated points
  for (const auto &point : triangulatedPoints) {
    std::cout << "Triangulated Point: " << point << std::endl;
  }

  /*
typedef Vec<float, 3> cv::Vec3f
see Vec

typedef Point3_<float> cv::Point3f

*/
}

int main() { projectPointcloudInStereoImagePlane(); }
