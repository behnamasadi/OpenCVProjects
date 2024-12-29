#include <opencv2/opencv.hpp>

void estimateFundamentalMatrix() {

  int numBoards = 0; // minimum 4
  int numCornersHor; // usually 9
  int numCornersVer; // usually 6
  float square_size; // in our case it is 0.025192

  numBoards = 10;
  numCornersHor = 9;
  numCornersVer = 6;
  square_size = 0.025192;

  int numSquares = numCornersHor * numCornersVer;
  cv::Size board_sz = cv::Size(numCornersHor, numCornersVer);

  std::vector<cv::Point2f> corners_left, corners_right;
  //    std::vector<cv::Point2f> left_imagePoints, right_imagePoints;
  std::vector<cv::KeyPoint> corners_left_keypoints, corners_right_keypoints;

  cv::Mat left_image, right_image, left_image_gray, right_image_gray;
  std::string left_image_path, right_image_path;
  left_image_path = "../../images/stereo_vision/left01.jpg";
  right_image_path = "../../images/stereo_vision/right01.jpg";

  left_image = cv::imread(left_image_path);
  right_image = cv::imread(right_image_path);

  cv::findChessboardCorners(right_image, board_sz, corners_right,
                            cv::CALIB_CB_ADAPTIVE_THRESH |
                                cv::CALIB_CB_FILTER_QUADS);
  cv::findChessboardCorners(left_image, board_sz, corners_left,
                            cv::CALIB_CB_ADAPTIVE_THRESH |
                                cv::CALIB_CB_FILTER_QUADS);

  cv::cvtColor(left_image, left_image_gray, cv::COLOR_RGB2GRAY);
  cv::cvtColor(right_image, right_image_gray, cv::COLOR_RGB2GRAY);

  cv::cornerSubPix(
      left_image_gray, corners_left, cv::Size(11, 11), cv::Size(-1, -1),
      cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 30,
                       0.1));
  cv::cornerSubPix(
      right_image_gray, corners_right, cv::Size(11, 11), cv::Size(-1, -1),
      cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 30,
                       0.1));

  std::vector<cv::DMatch> good_matches;
  for (std::size_t i = 0; i < corners_right.size(); i++) {
    cv::KeyPoint left_keypoint, right_keypoint;
    left_keypoint.pt = corners_left.at(i);
    right_keypoint.pt = corners_right.at(i);
    corners_left_keypoints.push_back(left_keypoint);
    corners_right_keypoints.push_back(right_keypoint);

    cv::DMatch match;
    match.imgIdx;
    match.queryIdx = i;
    match.trainIdx = i;
    good_matches.push_back(match);
  }

  //    for( int i = 0; i < descriptors_object.rows; i++ )
  //    { if( matches[i].distance < 1.5*min_dist )
  //        {
  //            good_matches.push_back( matches[i]);
  //        }
  //    }

  cv::Mat img_matches;
  cv::drawMatches(left_image, corners_left_keypoints, right_image,
                  corners_right_keypoints, good_matches, img_matches,
                  cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                  cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  cv::imwrite("matches.jpg", img_matches);

  cv::imshow("matches", img_matches);
  cv::waitKey(0);

  cv::Mat F =
      cv::findFundamentalMat(corners_left, corners_right, cv::FM_7POINT);
  std::cout << F << std::endl;

  cv::Mat left_image_cornor_points = cv::Mat(corners_left);
  cv::Mat right_image_cornor_points = cv::Mat(corners_right);

  std::vector<cv::Vec3f> left_lines, right_lines;

  cv::computeCorrespondEpilines(left_image_cornor_points, 1, F, right_lines);
  cv::computeCorrespondEpilines(right_image_cornor_points, 2, F, left_lines);

  for (std::size_t i = 0; i < right_lines.size(); i++) {

    std::cout << right_lines.at(i) << std::endl;
    cv::Vec3f l = right_lines.at(i);

    double a = l.val[0];
    double b = l.val[1];
    double c = l.val[2];

    /*ax+by+c=0*/
    double x0, y0, x1, y1;
    x0 = 0;
    y0 = (-c - a * x0) / b;

    std::cout << "x0: " << x0 << " y0: " << y0 << std::endl;

    x1 = right_image.cols;
    y1 = (-c - a * x1) / b;

    std::cout << "x1: " << x1 << " y1: " << y1 << std::endl;

    right_image.rows;
    right_image.cols;

    // cv::line(right_image, cvPoint(x0,y0),
    // cvPoint(right_image.cols,right_image.rows), cvScalar(0,255,0), 1);
    cv::line(right_image, cv::Point(x0, y0), cv::Point(x1, y1),
             cv::Scalar(0, 255, 0), 1);
    cv::imshow("right_image_epipolarline", right_image);
    cv::imwrite("right_image_epipolarline.png", right_image);

    cv::waitKey(0);
  }

  /*

     ┌u'1u1   u'1v1   u'1   v'1u1   v'1v1   v'1   u1   v1   1┐   ┌f11┐
     |u'2u2   u'2v2   u'2   v'2u2   v'2v2   v'2   u2   v2   1|   |f12|
     |u'3u3   u'3v3   u'3   v'3u3   v'3v3   v'3   u3   v3   1|   |f13|
     |u'4u4   u'4v4   u'4   v'4u4   v'4v4   v'4   u4   v4   1|   |f21|
     |u'5u5   u'5v5   u'5   v'5u5   v'5v5   v'5   u5   v5   1| * |f22|=0
     |u'6u6   u'6v6   u'6   v'6u6   v'6v6   v'6   u6   v6   1|   |f23|
     |u'7u7   u'7v7   u'7   v'7u7   v'7v7   v'7   u7   v7   1|   |f31|
     └u'8u8   u'8v8   u'8   v'8u8   v'8v8   v'8   u8   v8   1┘   |f32|
                                                                 └f33┘
  */

  cv::Mat left_cameraMatrix, right_cameraMatrix, left_distCoeffs,
      right_distCoeffs;
  left_cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
  right_cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

  cv::undistortPoints(left_image_cornor_points, left_image_cornor_points,
                      left_cameraMatrix, left_distCoeffs, cv::Mat(),
                      left_cameraMatrix);
  cv::undistortPoints(right_image_cornor_points, right_image_cornor_points,
                      right_cameraMatrix, right_distCoeffs, cv::Mat(),
                      right_cameraMatrix);

  //    cv::SVD svd;
  //    svd.

  /*
      CV_FM_7POINT
      CV_FM_8POINT
      CV_FM_RANSAC
      CV_FM_LMEDS
  */
  //    cv::Mat F= cv::findFundamentalMat(points1, points2, CV_FM_8POINT);
}

void findFundamentalMatrix(std::vector<cv::Point2d> &imagePointsLeftCamera,
                           std::vector<cv::Point2d> &imagePointsRightCamera) {
  std::vector<cv::Point3d> imagePointsLeftCameraHomogeneous,
      imagePointsRightCameraHomogeneous;
  cv::convertPointsToHomogeneous(imagePointsLeftCamera,
                                 imagePointsLeftCameraHomogeneous);
  cv::convertPointsToHomogeneous(imagePointsRightCamera,
                                 imagePointsRightCameraHomogeneous);
  /*

     ┌       ┐ ┌f11  f12  f13┐ ┌u┐
     |u` v` 1|*|f21  f22  f23|*|v|=0
     └       ┘ └f31  f32  f33┘ └1┘

     ┌u'1u1   u'1v1   u'1   v'1u1   v'1v1   v'1   u1   v1   1┐   ┌f11┐
     |u'2u2   u'2v2   u'2   v'2u2   v'2v2   v'2   u2   v2   1|   |f12|
     |u'3u3   u'3v3   u'3   v'3u3   v'3v3   v'3   u3   v3   1|   |f13|
     |u'4u4   u'4v4   u'4   v'4u4   v'4v4   v'4   u4   v4   1|   |f21|
     |u'5u5   u'5v5   u'5   v'5u5   v'5v5   v'5   u5   v5   1| * |f22|=0
     |u'6u6   u'6v6   u'6   v'6u6   v'6v6   v'6   u6   v6   1|   |f23|
     |u'7u7   u'7v7   u'7   v'7u7   v'7v7   v'7   u7   v7   1|   |f31|
     └u'8u8   u'8v8   u'8   v'8u8   v'8v8   v'8   u8   v8   1┘   |f32|
                                                                 └f33┘
  */
  double u_prime, v_prime, u, v;
  cv::Mat A = cv::Mat_<double>(imagePointsLeftCamera.size(), 9);
  for (std::size_t i = 0; i < imagePointsLeftCamera.size(); i++) {
    u_prime = imagePointsLeftCamera.at(i).x;
    v_prime = imagePointsLeftCamera.at(i).y;

    u = imagePointsRightCamera.at(i).x;
    v = imagePointsRightCamera.at(i).y;

    A.at<double>(i, 0) = u_prime * u;
    A.at<double>(i, 1) = u_prime * v;
    A.at<double>(i, 2) = u_prime;
    A.at<double>(i, 3) = v_prime * u;
    A.at<double>(i, 4) = v_prime * v;
    A.at<double>(i, 5) = v_prime;
    A.at<double>(i, 6) = u;
    A.at<double>(i, 7) = v;
    A.at<double>(i, 8) = 1;
  }

  cv::Mat U, SingularValuesVector, VT;
  cv::Mat SigmaMatrix = cv::Mat::zeros(A.rows, A.cols, CV_64F);
  cv::SVD::compute(A.clone(), SingularValuesVector, U, VT);

  //////////////////////////////////Building U (Building Square Matrix
  /// U)///////////////////////////////////

  cv::Mat completeU = cv::Mat_<double>(U.rows, U.rows);
  cv::Mat missingElementsOfU = cv::Mat::zeros(U.rows, U.rows - U.cols, CV_64F);
  cv::hconcat(U, missingElementsOfU, completeU);

  //////////////////////////////////Building Sigma Matrix
  //////////////////////////////////////

  cv::Mat completeSigma = cv::Mat::zeros(completeU.cols, VT.rows, CV_64F);
  for (int i = 0; i < SingularValuesVector.rows; i++) {
    completeSigma.at<double>(i, i) = SingularValuesVector.at<double>(i, 0);
  }

  //////////////////////////////////Checking A=completeU*completeSigma*Vt
  //////////////////////////////////////

  std::cout << "checking A-U*Sigma*VT=0" << std::endl;
  std::cout << cv::sum(A - completeU * completeSigma * VT).val[0] << std::endl;

  ///////////////////////////////////Building F Matrix From F vector
  ////////////////////////////////////////////////
  cv::Mat F_vec = VT.col(VT.cols - 1);
  std::cout << F_vec.cols << std::endl;
  cv::Mat F = cv::Mat(3, 3, cv::DataType<double>::type);

  F.at<double>(0, 0) = F_vec.at<double>(0, 0);
  F.at<double>(0, 1) = F_vec.at<double>(1, 0);
  F.at<double>(0, 2) = F_vec.at<double>(2, 0);
  F.at<double>(1, 0) = F_vec.at<double>(3, 0);
  F.at<double>(1, 1) = F_vec.at<double>(4, 0);
  F.at<double>(1, 2) = F_vec.at<double>(5, 0);
  F.at<double>(2, 0) = F_vec.at<double>(6, 0);
  F.at<double>(2, 1) = F_vec.at<double>(7, 0);
  F.at<double>(2, 2) = F_vec.at<double>(8, 0);

  ///////////////////////////////////Computing SVD of F
  ////////////////////////////////////////////////

  cv::SVD::compute(F.clone(), SingularValuesVector, U, VT);
  std::cout << "F singular values" << std::endl;
  std::cout << SingularValuesVector << std::endl;

  ///////////////////////////////////Setting The Smallest Eigen Value to
  /// Zero/////////////////////////////////////////////
  SingularValuesVector.at<double>(SingularValuesVector.rows - 1, 0) = 0;

  //////////////////////////////////Building U (Building Square Matrix
  /// U)///////////////////////////////////

  completeU = cv::Mat_<double>(U.rows, U.rows);
  missingElementsOfU = cv::Mat::zeros(U.rows, U.rows - U.cols, CV_64F);
  cv::hconcat(U, missingElementsOfU, completeU);

  //////////////////////////////////Building Sigma Matrix
  //////////////////////////////////////

  completeSigma = cv::Mat::zeros(completeU.cols, VT.rows, CV_64F);
  for (int i = 0; i < SingularValuesVector.rows; i++) {
    completeSigma.at<double>(i, i) = SingularValuesVector.at<double>(i, 0);
  }
  /////////////////////////////////////Building New F
  /// matrix///////////////////////////////////////

  cv::Mat NewF = completeU * completeSigma * VT;
  std::cout << "Fundamental Matrix is:" << std::endl;
  std::cout << NewF << std::endl;
}

int main(int argc, char **argv11) { estimateFundamentalMatrix(); }
