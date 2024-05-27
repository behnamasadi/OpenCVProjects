#include "transformation.hpp"
#include <opencv2/opencv.hpp>

void rodrigueRotationMatrix() {
  // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga61585db663d9da06b68e70cfbf6a1eac

  // Converts a rotation matrix to a rotation vector.

  cv::Mat rvec = cv::Mat_<double>(3, 1);
  cv::Mat rotationMatrix = cv::Mat_<double>(3, 3);

  double roll, pitch, yaw;
  roll = M_PI / 2;
  pitch = M_PI / 2;
  yaw = 0; // M_PI/6;

  rotationMatrix = rotationMatrixFromRollPitchYaw(roll, pitch, yaw);

  cv::Ptr<cv::Formatter> fmt = cv::Formatter::get(cv::Formatter::FMT_DEFAULT);

  fmt->set64fPrecision(3);
  fmt->set32fPrecision(3);

  std::cout << "Rotation Matrix:\n" << fmt->format(rotationMatrix) << std::endl;

  cv::Rodrigues(rotationMatrix, rvec);
  std::cout << "Calculated Rodrigues vector:\n" << rvec << std::endl;

  // Converts a rotation vector  to a rotation matrix to a or vice versa.
  rotationMatrix = cv::Mat::zeros(3, 3, CV_64FC1);
  cv::Rodrigues(rvec, rotationMatrix);
  std::cout << "Calculated Rotation Matrix from Rodrigues vector:\n"
            << rotationMatrix << std::endl;
}

int main(int argc, char *argv[]) {
  rodrigueRotationMatrix();
  return 0;
}
