#include <opencv2/opencv.hpp>
/*
    Harris operator
    Shi and Tomasi
    Level curve curvature
    Hessian feature strength measures
    SUSAN
    SIFT/SURF
    FAST
*/

// Shi-Tomasi Corner Detector & Good Features to Track
void shiTomasiGoodFeaturesToTrack(char **argv) {

  cv::Mat src = cv::imread(argv[1], cv::IMREAD_ANYCOLOR);
  cv::Mat src_gray;
  cv::cvtColor(src, src_gray, cv::COLOR_BGR2GRAY);

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
  cv::goodFeaturesToTrack(src_gray, corners, maxCorners, qualityLevel,
                          minDistance, cv::Mat(), blockSize, useHarrisDetector,
                          k);

  /// Draw corners detected
  std::cout << "** Number of corners detected: " << corners.size() << std::endl;
  int r = 4;
  for (std::size_t i = 0; i < corners.size(); i++) {
    cv::circle(src, corners[i], r,
               cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
                          rng.uniform(0, 255)),
               -1, 8, 0);
  }

  /// Show what you got
  std::string sourceWindow = "Image";
  cv::namedWindow(sourceWindow, cv::WINDOW_AUTOSIZE);
  cv::imshow(sourceWindow, src);
  cv::waitKey(0);
}

int main(int argc, char **argv) {

  if (argc == 1) {
    argv[1] = strdup("../images/building.jpg");
  }
  shiTomasiGoodFeaturesToTrack(argv);
  return 0;
}
