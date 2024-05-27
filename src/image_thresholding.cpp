#include <opencv2/opencv.hpp>

void thresholding(char **argv) {
  cv::Mat src_gray, dst;
  cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  /*
   0: Binary
   1: Binary Inverted
   2: Threshold Truncated
   3: Threshold to Zero
   4: Threshold to Zero Inverted
  */

  int threshold_value = 10;
  int threshold_type = 0;
  int const max_value = 255;
  int const max_type = 0;
  int const max_binary_value = 255;

  cv::threshold(src_gray, dst, threshold_value, max_binary_value,
                threshold_type);

  std::string namedWindow = "thresholding";
  cv::namedWindow(namedWindow, cv::WINDOW_AUTOSIZE);
  cv::imshow(namedWindow, dst);
  cv::waitKey(0);
}

int main(int argc, char **argv) {
  if (argc == 1) {
    argv[1] = strdup("../images/building.jpg");
  }
  thresholding(argv);
  return 0;
}
