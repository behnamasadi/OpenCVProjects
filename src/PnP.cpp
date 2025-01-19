#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>

template <typename T> void printArray(std::vector<T> array) {
  for (auto element : array)
    std::cout << element << std::endl;
}

void solvePerspectivenPoint(
    std::vector<cv::Vec3f> objectPointsInWorldCoordinate,
    cv::SolvePnPMethod method) {

  ///////// camera extrinsic parameters/////////

  cv::Mat cameraTranslation_w_c = (cv::Mat_<double>(3, 1) << 0, 0, 0);

  cv::Mat cameraRotation_w_c = cv::Mat::eye(3, 3, cv::DataType<double>::type);

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

  cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << focalLength * mx, 0, V0, 0,
                          focalLength * my, U0, 0, 0, 1);
  std::vector<cv::Point2f> imagePointsCamera;
  cv::projectPoints(objectPointsInWorldCoordinate, cameraRotation_w_c,
                    cameraTranslation_w_c, cameraMatrix, cv::noArray(),
                    imagePointsCamera);

  ///////// storing images from camera /////////

  int blue, green, red;
  blue = 255;
  green = 255;
  red = 255;

  std::string fileName;
  cv::Mat image;
  int U, V;
  image = cv::Mat::zeros(numberOfPixelInHeight, numberOfPixelInWidth, CV_8UC1);

  for (std::size_t i = 0; i < imagePointsCamera.size(); i++) {
    V = int(imagePointsCamera.at(i).x);
    U = int(imagePointsCamera.at(i).y);
    image.at<char>(U, V) = (char)255;

    // draw a circle at (U,V) with a radius of 20. Use green lines of width 5
    cv::circle(image, cv::Point(V, U), 5, cv::Scalar(blue, green, red), 3);
  }

  fileName = std::string("p3pCamera") + std::to_string(focalLength) +
             std::string("_.png");
  cv::imwrite(fileName, image);

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

/*
The cv::solvePnPGeneric() allows retrieving all the possible solutions.
Currently, only cv::SOLVEPNP_P3P, cv::SOLVEPNP_AP3P, cv::SOLVEPNP_IPPE,
cv::SOLVEPNP_IPPE_SQUARE, cv::SOLVEPNP_SQPNP can return multiple solutions.

*/

void PnPGeneric() {}

/*
The cv::solvePnPRansac() computes the object pose wrt. the camera frame using a
RANSAC scheme to deal with outliers.
*/

void PnPRansac() { // cv::solvePnPRansac()
}

void PnPPoseRefinement() {
  // solvePnPRefineLM()
  // solvePnPRefineVVS()
}

int main() {}
