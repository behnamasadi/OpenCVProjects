#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// We need 4 corresponding 2D points(x,y) to calculate homography.
vector<Point2f> left_image;  // Stores 4 points(x,y) of the logo image. Here the
                             // four points are 4 corners of image.
vector<Point2f> right_image; // stores 4 points that the user clicks(mouse left
                             // click) in the main image.

// Image containers for main and logo image
Mat imageMain;
Mat imageLogo;

// Function to add main image and transformed logo image and show final output.
// Icon image replaces the pixels of main image in this implementation.
void showFinal(Mat src1, Mat src2) {

  Mat gray, gray_inv, src1final, src2final;
  cvtColor(src2, gray, cv::COLOR_BGR2GRAY);
  threshold(gray, gray, 0, 255, cv::THRESH_BINARY);
  bitwise_not(gray, gray_inv);
  src1.copyTo(src1final, gray_inv);
  src2.copyTo(src2final, gray);
  Mat finalImage = src1final + src2final;
  namedWindow("output", WINDOW_AUTOSIZE);
  imshow("output", finalImage);
  cv::waitKey(0);
}

// Here we get four points from the user with left mouse clicks.
// On 5th click we output the overlayed image.
void on_mouse(int e, int x, int y, int d, void *ptr) {
  if (e == EVENT_LBUTTONDOWN) {
    if (right_image.size() < 4) {

      right_image.push_back(Point2f(float(x), float(y)));
      cout << x << " " << y << endl;
    } else {
      cout << " Calculating Homography " << endl;
      // Deactivate callback
      cv::setMouseCallback("Display window", NULL, NULL);
      // once we get 4 corresponding points in both images calculate homography
      // matrix
      Mat H = findHomography(left_image, right_image, 0);
      Mat logoWarped;
      // Warp the logo image to change its perspective
      warpPerspective(imageLogo, logoWarped, H, imageMain.size());
      showFinal(imageMain, logoWarped);
    }
  }
}

/*
How to run:
./main  ../images/homography/main.jpg ../images/homography/logo.jpg
*/
int main(int argc, char **argv) {
  //  We need tow argumemts. "Main image" and "logo image"
  if (argc != 3) {
    cout << " Usage: error, please provide  <main.jpg>  <logo.jpg> " << endl;
    return -1;
  }

  // Load images from arguments passed.
  imageMain = imread(argv[1], cv::IMREAD_COLOR);
  imageLogo = imread(argv[2], cv::IMREAD_COLOR);
  // Push the 4 corners of the logo image as the 4 points for correspondence to
  // calculate homography.
  left_image.push_back(Point2f(float(0), float(0)));
  left_image.push_back(Point2f(float(0), float(imageLogo.rows)));
  left_image.push_back(Point2f(float(imageLogo.cols), float(imageLogo.rows)));
  left_image.push_back(Point2f(float(imageLogo.cols), float(0)));

  namedWindow("Display window",
              WINDOW_AUTOSIZE); // Create a window for display.
  imshow("Display window", imageMain);

  setMouseCallback("Display window", on_mouse, NULL);

  //  Press "Escape button" to exit
  while (1) {
    int key = cv::waitKey(10);
    if ((char)key == (char)27)
      break;
  }

  return 0;
}
