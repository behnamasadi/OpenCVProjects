#include "collection_adapters.hpp"
#include "transformation.hpp"
#include <opencv2/opencv.hpp>
#include <rerun.hpp>
#include <rerun/archetypes/arrows3d.hpp>

template <typename T> void printArray(std::vector<T> array) {
  for (auto element : array)
    std::cout << element << std::endl;
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
      // draw a circle at (U,V) with a radius of 20. Use green lines of width 5
      cv::circle(cameraImage, cv::Point(int(col), int(row)), 5,
                 cv::Scalar(blue, green, red), 3);

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

void p3p() {

  const auto rec = rerun::RecordingStream("p3p");
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

  ///////// creating points in the world coordinate /////////
  std::vector<cv::Vec3f> objectPointsInWorldCoordinate = {
      {-0.5, 0.5, 1}, {-0.2, 0.8, 0.5}, {0.8, 0, 0.7}};

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

  ///////// camera extrinsic parameters/////////

  cv::Mat cameraTranslation_w_c = (cv::Mat_<double>(3, 1) << 0, 0, 0);

  cv::Mat cameraRotation_w_c = cv::Mat::eye(3, 3, cv::DataType<double>::type);

  // Log the data
  std::string camera_name = "world/camera";

  ///////// camera intrinsic parameters/////////

  uint32_t numberOfPixelInHeight, numberOfPixelInWidth;
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

  cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << focalLength * mx, 0, V0, 0,
                          focalLength * my, U0, 0, 0, 1);
  std::vector<cv::Point2f> imagePointsCamera;
  cv::projectPoints(objectPointsInWorldCoordinate, cameraRotation_w_c,
                    cameraTranslation_w_c, cameraMatrix, cv::noArray(),
                    imagePointsCamera);

  ///////// storing images from camera /////////

  rec.log(camera_name,
          rerun::Pinhole::from_focal_length_and_resolution(
              {float(focalLength * mx), float(focalLength * my)},
              {float(numberOfPixelInWidth), float(numberOfPixelInHeight)}));

  // Extract translation vector
  rerun::Vec3D translation =
      getRerunTranslationFromCvMat<double>(cameraTranslation_w_c);

  rerun::Mat3x3 rotation_matrix =
      getRerunRotationFromCvMat<double>(cameraRotation_w_c);

  rec.log(camera_name, rerun::Transform3D(translation, rotation_matrix));

  std::string fileName;

  int U, V;

  fileName = std::string("p3pCamera") + std::to_string(focalLength) +
             std::string("_.png");

  cv::Mat image =
      createImage(focalLength, numberOfPixelInHeight, numberOfPixelInWidth,
                  imagePointsCamera, fileName);

  // Log the image to the camera entity in the hierarchy
  rec.log("world/camera/image/rgb",
          rerun::Image::from_greyscale8(
              image, {numberOfPixelInWidth, numberOfPixelInHeight}));

  ///////// solveP3P /////////
  std::vector<cv::Mat> tvecs, rvecs;

  int flags = cv::SOLVEPNP_P3P; // or SOLVEPNP_AP3P

  cv::solveP3P(objectPointsInWorldCoordinate, imagePointsCamera, cameraMatrix,
               cv::noArray(), rvecs, tvecs, flags);

  //  The solutions are sorted by reprojection errors (lowest to highest).
  std::cout << "Possible Rotations:" << std::endl;

  printArray(rvecs);
  std::cout << "Possible Translations:" << std::endl;
  printArray(tvecs);
}

int main() { p3p(); }
