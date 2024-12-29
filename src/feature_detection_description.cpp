#include "arguments_parser.hpp"
#include <opencv2/opencv.hpp>

// Shi-Tomasi Corner Detector & Good Features to Track
void shiTomasiGoodFeaturesToTrack(cv::Mat img) {

  cv::Mat img_gray;

  // Check if the image has 3 channels (color)
  if (img.channels() == 3) {
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);
    std::cout << "image has 3 channel" << std::endl;

  } else {
    std::cout << "image is already gray scale" << std::endl;
    img_gray = img.clone();
  }

  int maxCorners = 100;

  cv::RNG rng(12345);

  /// Parameters for Shi-Tomasi algorithm
  std::vector<cv::Point2f> corners;
  double qualityLevel = 0.01;
  double minDistance = 10;
  int blockSize = 3;
  bool useHarrisDetector = false;
  double k = 0.04;

  /// Apply corner detection
  cv::goodFeaturesToTrack(img_gray, corners, maxCorners, qualityLevel,
                          minDistance, cv::Mat(), blockSize, useHarrisDetector,
                          k);

  /// Draw corners detected
  std::cout << "** Number of corners detected: " << corners.size() << std::endl;
  int r = 4;
  cv::Mat img_color;
  cv::cvtColor(img, img_color, cv::COLOR_GRAY2BGR);

  for (std::size_t i = 0; i < corners.size(); i++) {
    cv::circle(img_color, corners[i], r,
               cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
                          rng.uniform(0, 255)),
               -1, 8, 0);
  }

  // Show what you got
  std::string sourceWindow = "goodFeaturesToTrack";
  cv::namedWindow(sourceWindow, cv::WINDOW_AUTOSIZE);
  cv::imshow(sourceWindow, img_color);
  cv::imwrite("goodFeaturesToTrack.png", img_color);
  cv::waitKey(0);
}

void keyPointDetectorDescriptor(cv::Mat img) {

  // Create SIFT (floating-point descriptors)
  cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
  std::cout << sift->getDefaultName() << std::endl;
  // Check descriptor type
  int descType = sift->descriptorType(); // Returns CV_32F, <=>5
  std::cout << "SIFT Descriptor Type: " << descType << " (CV_32F = " << CV_32F
            << ")" << std::endl;

  // Create ORB (binary descriptors)
  cv::Ptr<cv::ORB> orb = cv::ORB::create();
  std::cout << orb->getDefaultName() << std::endl;

  // Check descriptor type
  descType = orb->descriptorType(); // Returns CV_8U
  std::cout << "ORB Descriptor Type: " << descType << " (CV_8U = " << CV_8U
            << ")" << std::endl;

  std::vector<cv::KeyPoint> k_pts;

  sift->detect(img, k_pts);
  // Draw keypoints on the original image
  cv::Mat imgWithSIFTKeypoints;
  cv::drawKeypoints(img, k_pts, imgWithSIFTKeypoints, cv::Scalar::all(-1),
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  for (int i = 0; i < 5; i++) {

    std::cout << "angle: " << k_pts[i].angle << std::endl;
    std::cout << "class_id:" << k_pts[i].class_id << std::endl;
    std::cout << "pt:" << k_pts[i].pt << std::endl;
  }

  std::vector<cv::Point2f> points;
  cv::KeyPoint::convert(k_pts, points);

  // Display the result
  cv::imshow("SIFT Keypoints", imgWithSIFTKeypoints);
  cv::imwrite("SIFTKeypoints.png", imgWithSIFTKeypoints);
  cv::waitKey(0); // Wait for a key press

  ///////////////////////////////// ORB //////////////////////////////

  k_pts.clear();

  orb->detect(img, k_pts);
  // Draw keypoints on the original image
  cv::Mat imgWithORBKeypoints;
  cv::drawKeypoints(img, k_pts, imgWithORBKeypoints, cv::Scalar::all(-1),
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  for (int i = 0; i < 5; i++) {

    std::cout << "angle: " << k_pts[i].angle << std::endl;
    std::cout << "class_id:" << k_pts[i].class_id << std::endl;
    std::cout << "pt:" << k_pts[i].pt << std::endl;
  }

  // Display the result
  cv::imshow("ORB Keypoints", imgWithORBKeypoints);
  cv::imwrite("ORBKeypoints.png", imgWithORBKeypoints);
  cv::waitKey(0); // Wait for a key press
}

int main(int argc, char **argv) {

  ArgumentsParser input(argc, argv);
  std::string image_path =
      "../../images/feature_detection_description/building.jpg";

  std::string inputFilename = input.getArg("-i");
  if (inputFilename.empty()) {
    std::cerr << "no intput file is given, default will be used: " << image_path
              << std::endl;
  } else {
    image_path = inputFilename;
  }

  cv::Mat img = cv::imread(image_path, cv::IMREAD_ANYCOLOR);

  keyPointDetectorDescriptor(img);
  shiTomasiGoodFeaturesToTrack(img);

  return 0;
}
