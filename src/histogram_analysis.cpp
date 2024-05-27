#include <opencv2/opencv.hpp>

/*
https://en.wikipedia.org/wiki/Histogram_equalization
https://en.wikipedia.org/wiki/Cumulative_distribution_function#Inverse_distribution_function_(quantile_function)
https://docs.opencv.org/3.4/d4/d1b/tutorial_histogram_equalization.html
*/

void imageNormalization(char **argv) {
  cv::Mat src_img = imread(argv[1], cv::IMREAD_COLOR);
  cv::Mat normalized_src_img, converted_src_img;

  double alpha = 1.0 / 255, beta = 0;

  // alpha*old_pixel+ beta
  src_img.convertTo(converted_src_img, CV_64FC3, alpha, beta);

  double lower_range_boundary, upper_range_boundary;
  lower_range_boundary = 0;
  upper_range_boundary = 1;

  cv::normalize(src_img, normalized_src_img, lower_range_boundary,
                upper_range_boundary, cv::NORM_MINMAX, CV_64FC3);
  cv::imshow("Image converted", converted_src_img);
  cv::imshow("Image Normalized", normalized_src_img);
  cv::waitKey(0);
}

void histogramCalculation(char **argv) {
  using namespace cv;
  using namespace std;

  cv::Mat src = cv::imread(argv[1], cv::IMREAD_COLOR);

  vector<Mat> bgr_planes;
  split(src, bgr_planes);

  // bins: It is the number of subdivisions in each dim.
  int bins = 256;

  // intensity in the range 0−255), the upper boundary is exclusive
  float range[] = {0, 256};
  const float *histRange[] = {range};
  bool uniform = true, accumulate = false;
  Mat b_hist, g_hist, r_hist, b_hist_normalized, g_hist_normalized,
      r_hist_normalized;

  calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &bins, histRange, uniform,
           accumulate);
  calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &bins, histRange, uniform,
           accumulate);
  calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &bins, histRange, uniform,
           accumulate);

  int hist_w = 512, hist_h = 400;
  int bin_w = cvRound((double)hist_w / bins);

  Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

  Mat histImageNormalized(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

  normalize(b_hist, b_hist_normalized, 0, histImage.rows, NORM_MINMAX, -1,
            Mat());
  normalize(g_hist, g_hist_normalized, 0, histImage.rows, NORM_MINMAX, -1,
            Mat());
  normalize(r_hist, r_hist_normalized, 0, histImage.rows, NORM_MINMAX, -1,
            Mat());
  for (int i = 1; i < bins; i++) {

    line(histImageNormalized,
         Point(bin_w * (i - 1),
               hist_h - cvRound(b_hist_normalized.at<float>(i - 1))),
         Point(bin_w * (i), hist_h - cvRound(b_hist_normalized.at<float>(i))),
         Scalar(255, 0, 0), 2, 8, 0);
    line(histImageNormalized,
         Point(bin_w * (i - 1),
               hist_h - cvRound(g_hist_normalized.at<float>(i - 1))),
         Point(bin_w * (i), hist_h - cvRound(g_hist_normalized.at<float>(i))),
         Scalar(0, 255, 0), 2, 8, 0);
    line(histImageNormalized,
         Point(bin_w * (i - 1),
               hist_h - cvRound(r_hist_normalized.at<float>(i - 1))),
         Point(bin_w * (i), hist_h - cvRound(r_hist_normalized.at<float>(i))),
         Scalar(0, 0, 255), 2, 8, 0);
  }
  imshow("Source image", src);

  imshow("calcHist Normalized", histImageNormalized);
  waitKey();
}

void histogramCalculationNewAppraoch(char **argv) {

  // https://stackoverflow.com/questions/42902527/how-to-set-the-parameters-of-opencv3-calchist-using-vectors
  cv::Mat3b src = cv::imread(argv[1]);

  //    cv::Mat3b hsv;
  //    cv::cvtColor(src, hsv,cv::COLOR_BGR2HSV);

  cv::Mat oldApproachHist, newApproachHist;

  {
    // Quantize the each channel to 30 levels

    int b_bins = 30, g_bins = 30, r_bins = 30;
    int bins[] = {b_bins, g_bins, r_bins};

    // the upper boundary is exclusive

    float b_ranges[] = {0, 256};
    float g_ranges[] = {0, 256};
    float r_ranges[] = {0, 256};

    const float *ranges[] = {b_ranges, g_ranges, r_ranges};

    // we compute the histogram from the 0-th,  1-st  and 2-nd channels
    int channels[] = {0, 1, 2};
    calcHist(&src, 1, channels, cv::Mat(), // do not use mask
             oldApproachHist, 3, bins, ranges,
             true, // the histogram is uniform
             false);
  }

  std::vector<cv::Mat> bgr_hist;

  cv::split(oldApproachHist, bgr_hist);

  cv::Mat b_hist, g_hist, r_hist;

  int bins = 256;

  int hist_w = 512, hist_h = 400;
  int bin_w = cvRound((double)hist_w / bins);
  cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1,
                cv::Mat());
  cv::normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1,
                cv::Mat());
  cv::normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1,
                cv::Mat());

  std::cout << oldApproachHist.dims << std::endl;
  std::cout << oldApproachHist.rows << std::endl;
  std::cout << oldApproachHist.cols << std::endl;
  std::cout << oldApproachHist.channels() << std::endl;
}

void histogramComparison() {
  // Histogram comparison methods
  /*

  https://docs.opencv.org/3.4/d6/dc7/group__imgproc__hist.html#ga994f53817d621e2e4228fc646342d386
  */
}

void histogramMatching() {}

void contrastStretching(char **argv) {
  // https://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm
}

void histogramEqualization(int argc, char **argv) {
  // https://stackoverflow.com/questions/41118808/difference-between-contrast-stretching-and-histogram-equalization
  cv::Mat src_img = imread(argv[1], cv::IMREAD_GRAYSCALE);
  cv::Mat equalized_img;
  src_img.copyTo(equalized_img);
  cv::equalizeHist(src_img, equalized_img);
  cv::imshow("Equlized Histogram Image", equalized_img);
  cv::imshow("Original Image", src_img);
  cv::waitKey(0);
}

int main(int argc, char **argv) {

  if (argc == 1) {
    argv[1] = strdup("../images/lena.jpg");
  }

  imageNormalization(argv);
}
