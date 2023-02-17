#include <opencv2/opencv.hpp>

/*Blob detection

Laplacian of Gaussian (LoG)
Difference of Gaussians (DoG)
Determinant of Hessian (DoH)
Maximally stable extremal regionsPCBR
*/

void smoothingBlurFilter(int argc, char **argv) {
  if (argc != 2) {
    std::cout << " Usage: ./main <image> \n" << std::endl;
    return;
  }

  cv::Mat src;
  src = cv::imread(argv[1], cv::IMREAD_ANYCOLOR);
  cv::Mat dst;
  dst = src.clone();

  int text_row, text_col;

  text_row = 10;
  text_col = 50;

  int DELAY_BLUR = 100;
  int MAX_KERNEL_LENGTH = 31;
  int DELAY_CAPTION = 1500;
  //	Point(-1, -1): Indicates where the anchor point (the pixel evaluated) is
  //located with respect to the neighborhood. 	If there is a negative value, then
  //the center of the kernel is considered the anchor point.

  cv::namedWindow("Filter Demo", cv::WINDOW_AUTOSIZE);
  cv::putText(src, "Original Image", cv::Point(text_row, text_col),
              cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 185, 0), 2);

  imshow("Filter Demo", dst);
  cv::waitKey(DELAY_CAPTION);

  // Applying Homogeneous blur
  cv::putText(src, "Homogeneous Blur (Average Fiter)",
              cv::Point(text_row, text_col), cv::FONT_HERSHEY_DUPLEX, 1.0,
              CV_RGB(118, 185, 0), 2);

  for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2) {
    blur(src, dst, cv::Size(i, i), cv::Point(-1, -1));
    imshow("Filter Demo", dst);
    cv::waitKey(DELAY_BLUR);
  }

  // Applying Gaussian blur
  cv::putText(src, "Gaussian", cv::Point(text_row, text_col),
              cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 185, 0), 2);

  for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2) {
    //		sigmax The standard deviation in x. 0 Writing  implies that  is
    //calculated using kernel size. 		sigmay The standard deviation in y. 0
    //Writing  implies that  is calculated using kernel size.

    //		c style:
    GaussianBlur(src, dst, cv::Size(i, i), 0, 0);
    imshow("Filter Demo", dst);
    cv::waitKey(DELAY_BLUR);
  }

  // Applying Median blur
  cv::putText(src, "Median Blur", cv::Point(text_row, text_col),
              cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 185, 0), 2);

  for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2) {
    medianBlur(src, dst, i);
    imshow("Filter Demo", dst);
    cv::waitKey(DELAY_BLUR);
  }

  // Applying Bilateral Filter
  cv::putText(src, "Bilateral Blur", cv::Point(text_row, text_col),
              cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 185, 0), 2);

  for (int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2) {
    bilateralFilter(src, dst, i, i * 2, i / 2);
    imshow("Filter Demo", dst);
    cv::waitKey(DELAY_BLUR);
  }

  // Wait until user press a key
  cv::putText(src, "End: Press a key!", cv::Point(text_row, text_col),
              cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(118, 185, 0), 2);

  cv::waitKey(0);
  return;
}

// Laplacian is based on second derivative
void laplacianOfGaussiansEdgeDetector(int argc, char **argv) {
  if (argc != 2) {
    std::cout << " Usage: ./main <image> \n" << std::endl;
    return;
  }
  cv::Mat src, src_gray, dst;
  int kernel_size = 3;
  int scale = 1;
  int delta = 0;
  //  ddepth: Depth of the destination image. Since our input is CV_8U we define
  //  ddepth = CV_16S to avoid overflow
  int ddepth = CV_16S;
  const char *window_name = "Laplace Demo";

  int c;

  // Load an image
  src = cv::imread(argv[1], cv::IMREAD_ANYCOLOR);

  // Remove noise by blurring with a Gaussian filter
  cv::GaussianBlur(src, src, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

  // Convert the image to grayscale
  cvtColor(src, src_gray, cv::COLOR_RGB2GRAY);

  // Create window
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

  // Apply Laplace function
  cv::Mat abs_dst;

  Laplacian(src_gray, dst, ddepth, kernel_size, scale, delta,
            cv::BORDER_DEFAULT);
  convertScaleAbs(dst, abs_dst);

  // Show what you got
  cv::imshow(window_name, abs_dst);

  cv::waitKey(0);
}

// DoG is approximation of Log which is based on second derivative
void DifferenceOfGaussians(int argc, char **argv) {
  if (argc != 2) {
    std::cout << " Usage: ./main <image> \n" << std::endl;
    return;
  }
  cv::Mat src, src_gray, dst, dog_1, dog_2;
  int kernel_size1 = 3;
  int kernel_size2 = 5;
  int scale = 1;
  int delta = 0;
  const char *window_name = "DoG Demo";

  int c;

  // Load an image
  src = cv::imread(argv[1], cv::IMREAD_ANYCOLOR);

  dog_1 = src.clone();
  dog_2 = src.clone();

  int invert;
  GaussianBlur(src, dog_2, cv::Size(kernel_size1, kernel_size1), 0, 0);
  GaussianBlur(src, dog_1, cv::Size(kernel_size2, kernel_size2), 0, 0);
  // cvSub(dog_2, dog_1, dst, 0);
  return;
}

int main(int argc, char **argv) {
  // smoothingBlurFilter( argc,  argv);
  laplacianOfGaussiansEdgeDetector(argc, argv);
  return 0;
}
