#include <opencv2/opencv.hpp>
// https://learnopencv.com/image-filtering-using-convolution-in-opencv/
// https://learnopencv.com/image-filtering-using-convolution-in-opencv/#intro-convo-kernels
void flipingKernel() {
  cv::Mat kernel = (cv::Mat_<double>(3, 3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
  std::cout << "kernel:\n" << kernel << std::endl;

  cv::Mat upside_down, mirrored;

  int flipCode;
  /*
  0 means flipping around the x-axis
  Positive value means flipping around y-axis.
  Negative value means flipping around both axes.
  */
  flipCode = 0;
  cv::flip(kernel, upside_down, flipCode);
  std::cout << "upside down:\n" << upside_down << std::endl;

  flipCode = 1;
  cv::flip(upside_down, mirrored, flipCode);
  std::cout << "mirrored:\n" << mirrored << std::endl;
}

void separableLinearFilter() {
  /*
  The function applies a separable linear filter to the image. That is, first,
  every row of src is filtered with the 1D kernel kernelX. Then, every column of
  the result is filtered with the 1D kernel kernelY. The final result shifted by
  delta is stored in dst .
  */
  // sepFilter2D();
}

void conv1D(char **argv) {

  cv::Mat src_img = imread(argv[1], cv::IMREAD_COLOR);
  cv::Mat kernel_x;
  // kernel_x<<1;

  // The function does actually compute correlation, not the convolution!
  // cv::filter2D();
}

int main(int argc, char **argv) {
  if (argc == 1) {
    argv[1] = strdup("../images/lena.jpg");
  }
  flipingKernel();
}
