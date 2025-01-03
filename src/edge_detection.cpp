#include <opencv2/opencv.hpp>
// https://learnopencv.com/edge-detection-using-opencv/

using namespace cv;
/*
Edge detection:

Canny
Deriche
Differential
Sobel
PrewittRoberts
Cross
*/

void prewitt(char **argv) {
  cv::Mat G_x = (cv::Mat_<double>(3, 3) << +1, 0, -1, +1, 0, -1, +1, 0, -1);
  cv::Mat G_y = (cv::Mat_<double>(3, 3) << +1, +1, +1, 0, 0, 0, -1, -1, -1);

  std::cout << "G_x:\n" << G_x << std::endl;

  cv::Mat src_img = imread(argv[1], cv::IMREAD_GRAYSCALE);
  cv::Mat src_img_G_x, src_img_G_y;

  cv::Mat upside_down, mirrored;

  int flipCode;
  /*
  0 means flipping around the x-axis
  Positive value means flipping around y-axis.
  Negative value means flipping around both axes.
  */
  flipCode = 0;
  cv::flip(G_x, upside_down, flipCode);
  std::cout << "G_x upside down:\n" << upside_down << std::endl;

  flipCode = 1;
  cv::flip(upside_down, mirrored, flipCode);
  std::cout << "G_x upside downed mirrored:\n" << mirrored << std::endl;

  cv::filter2D(src_img, src_img_G_x, -1, mirrored);

  std::cout << "G_y:\n" << G_y << std::endl;

  flipCode = 0;
  cv::flip(G_y, upside_down, flipCode);
  std::cout << "G_y upside down:\n" << upside_down << std::endl;

  flipCode = 1;
  cv::flip(upside_down, mirrored, flipCode);
  std::cout << "G_y upside downed mirrored:\n" << mirrored << std::endl;

  cv::filter2D(src_img, src_img_G_y, -1, mirrored);

  cv::imshow("G_x", src_img_G_x);
  cv::imshow("G_y", src_img_G_y);
  cv::waitKey(0);
}

// https://www.youtube.com/channel/UCf0WB91t8Ky6AuYcQV0CcLw/videos
// https://www.youtube.com/watch?v=7AlwDYmjrcs
// https://en.wikipedia.org/wiki/Roberts_cross
// https://en.wikipedia.org/wiki/Edge_detection#Differential

// Canny is based on first derivative

/// Global variables

cv::Mat canny_edge_src, canny_edge_src_gray;
cv::Mat canny_edge_dst, canny_edge_detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
std::string window_name = "Edge Map";

void CannyEdgeDetector(int, void *) {
  /*
  Step 1:
      Apply a Gaussian blur
  Step 2:
      Find edge gradient strength and angle
  Step 3:
      classify direction of gradient into 0,1,2,3
      round the gradient direction theta to nearest 45 degree
      i.e if gradient angle is 23 degree it will fall into 0 interval

  Step 4:
      Suppress non-maximum edges
      for a 3x3 neighbor of each pixel
      select two neigbors which are in the direction of gradient, i.e if the
  direction of gradient at point x is 1, we select the neighbor at top right and
      bottom left:

      3 2 1
      0 x 0
      1 2 3

      and if the magnitude of gradient is biger than both keep it otherwise set
  it to zero

  Step 5:
      Hysteresis thresholding: use of two thresholds, a high   HT and a low LT:
      pixel value in non-maxima suppressed image M (x,y) greater than HT is
  immediately accepted as an edge pixel pixel value in non-maxima suppressed
  image M (x,y) below the LT is immediately rejected. pixels in M (x,y) whose
  values lie between the two thresholds are accepted if they are connected to
  the already detected edge pixels
  */

  /// Reduce noise with a kernel 3x3
  blur(canny_edge_src_gray, canny_edge_detected_edges, Size(3, 3));

  /// Canny detector
  cv::Canny(canny_edge_detected_edges, canny_edge_detected_edges, lowThreshold,
            lowThreshold * ratio, kernel_size);

  /// Using Canny's output as a mask, we display our result
  canny_edge_dst = Scalar::all(0);

  canny_edge_src.copyTo(canny_edge_dst, canny_edge_detected_edges);
  imshow(window_name, canny_edge_dst);
}

int CannyEdgeDetector_Test(char **argv) {
  /// Load an image
  canny_edge_src = imread(argv[1]);

  if (!canny_edge_src.data) {
    return -1;
  }

  /// Create a matrix of the same type and size as src (for dst)
  canny_edge_dst.create(canny_edge_src.size(), canny_edge_src.type());

  /// Convert the image to grayscale
  cvtColor(canny_edge_src, canny_edge_src_gray, cv::COLOR_BGR2GRAY);

  /// Create a window
  namedWindow(window_name, cv::WINDOW_AUTOSIZE);

  /// Create a Trackbar for user to enter threshold
  createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold,
                 CannyEdgeDetector);

  /// Show the image
  CannyEdgeDetector(0, 0);

  /// Wait until user exit program by pressing a key
  waitKey(0);

  return 0;
}

// Sobel is based on first derivative so we have to set a threshold value after
// applying the algorithm to threshold the edges
void sobelEdgeDetector() {
  Mat src_gray;
  Mat grad;

  // Create window
  std::string window_name = "Sobel Edge Detector";
  namedWindow(window_name, cv::WINDOW_AUTOSIZE);

  int scale = 1;
  int delta = 0;

  // If src is 8-bit then the dst must be of depth 16S to avoid overflow.
  int depth = CV_16S;

  // Load an image
  std::string imgFileName = std::string("../images/lena.jpg");
  src_gray = imread(imgFileName, cv::IMREAD_GRAYSCALE);

  GaussianBlur(src_gray, src_gray, Size(3, 3), 0, 0, BORDER_DEFAULT);

  /// Generate grad_x and grad_y
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  int dx_order;
  int dy_order;
  /*
     kernel_size=aperture_size
     The aperture_size parameter should be odd and is the width (and the height)
     of the square fi lter. Currently, aperture_sizes of 1, 3, 5, and 7 are
     supported. The larger kernels give a better approximation to the derivative
     because the smaller kernels are very sensitive to noise.
  */
  int kernel_size = 3;

  /// Gradient X
  // Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  dx_order = 1;
  dy_order = 0;
  Sobel(src_gray, grad_x, depth, dx_order, dy_order, kernel_size, scale, delta,
        BORDER_DEFAULT);
  // because we can't display negative values so we have to convert the scale
  convertScaleAbs(grad_x, abs_grad_x);

  /// Gradient Y
  dx_order = 0;
  dy_order = 1;
  // Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  Sobel(src_gray, grad_y, depth, dx_order, dy_order, kernel_size, scale, delta,
        BORDER_DEFAULT);
  // because we can't display negative values so we have to convert the scale
  convertScaleAbs(grad_y, abs_grad_y);

  /// Total Gradient (approximate)
  // TotalGradient=sqrt( grad_y^2 + grad_t^2 ) ==>approximate  =0.5*abs_grad_x+
  // 0.5*abs_grad_y
  addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

  imshow(window_name, grad);

  waitKey(0);
}

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
  // located with respect to the neighborhood. 	If there is a negative value,
  // then the center of the kernel is considered the anchor point.

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
    // calculated using kernel size. 		sigmay The standard deviation in y.
    // 0 Writing  implies that  is calculated using kernel size.

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

int main1(int argc, char **argv) {
  // smoothingBlurFilter( argc,  argv);
  laplacianOfGaussiansEdgeDetector(argc, argv);
  return 0;
}

int main(int argc, char **argv) {
  // sobelEdgeDetector();
  if (argc == 1) {
    argv[1] = strdup("../images/lena.jpg");
  }
  prewitt(argv);
  return 0;
}
