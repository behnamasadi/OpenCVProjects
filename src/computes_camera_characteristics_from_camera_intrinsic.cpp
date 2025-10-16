// https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga87955f4330d5c20e392b265b7f92f691
#include <opencv2/opencv.hpp>
int main() {

  ///////////////// camera intrinsic /////////////////
  int numberOfPixelInHeight, numberOfPixelInWidth;
  double heightOfSensor, widthOfSensor;
  // double focalLength = 0.1;
  double focalLength = 2.0;
  double mx, my, cx, cy;

  numberOfPixelInHeight = 480;
  numberOfPixelInWidth = 640;

  heightOfSensor = 10;
  widthOfSensor = 10;

  mx = (numberOfPixelInWidth) / widthOfSensor;
  my = (numberOfPixelInHeight) / heightOfSensor;

  cx = (numberOfPixelInWidth) / 2;
  cy = (numberOfPixelInHeight) / 2;

  cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << focalLength * mx, 0, cx, 0,
                          focalLength * my, cy, 0, 0, 1);


  double  	fovx;
  double  	fovy;
  double 	calculatedFocalLength;
  cv::Point2d  	principalPoint;
  double  	aspectRatio ;

//  imageSize	Input image size in pixels.
//  apertureWidth	Physical width in mm of the sensor.
//  apertureHeight	Physical height in mm of the sensor.


  cv::calibrationMatrixValues(cameraMatrix,cv::Size(numberOfPixelInWidth,numberOfPixelInHeight),widthOfSensor,heightOfSensor,fovx,fovy,calculatedFocalLength,principalPoint,aspectRatio);

  std::cout<<"fovx: "<<fovx<<
  " fovy: " <<fovy<<
  " calculatedFocalLength: " <<calculatedFocalLength<<
  " principalPoint: "<< principalPoint<<
   " aspectRatio: "<<aspectRatio <<std::endl;


}
