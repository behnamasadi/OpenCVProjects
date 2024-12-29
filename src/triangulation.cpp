// http://www.tobias-weis.de/triangulate-3d-points-from-3d-imagepoints-from-a-moving-camera/

#include "arguments_parser.hpp"
#include <opencv2/opencv.hpp>
#include <rerun.hpp>

void correspondencesMatching(cv::Mat img1, cv::Mat img2) {

  // Create SIFT (floating-point descriptors)
  cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;

  sift->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
  sift->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);

  //////////////////////// BFMatcher,knnMatches ////////////////////////////

  cv::BFMatcher matcher(cv::NORM_L2);
  std::vector<cv::DMatch> matches;
  std::vector<std::vector<cv::DMatch>> knnMatches;
  int k = 2;
  matcher.knnMatch(descriptors1, descriptors2, knnMatches, k);

  // Apply Lowe's ratio test to filter matches
  const float ratioThresh = 0.75f; // Lowe's ratio test threshold
  std::vector<cv::DMatch> goodMatches;
  for (const auto &knnMatch : knnMatches) {
    if (knnMatch.size() >= 2 &&
        knnMatch[0].distance < ratioThresh * knnMatch[1].distance) {
      goodMatches.push_back(knnMatch[0]);
    }
  }

  cv::Mat img_knn_matches;

  cv::drawMatches(img1, keypoints1, img2, keypoints2, goodMatches,
                  img_knn_matches);
  cv::imshow("KNN Matches", img_knn_matches);

  // Define the matrix as a cv::Mat
  //   cv::Mat K = (cv::Mat_<double>(3, 3) << 707.0912, 0.0, 601.8873, 0.0,
  //   707.0912,
  //                183.1104, 0.0, 0.0, 1.0);

  cv::Mat K = (cv::Mat_<double>(3, 3) << 8.4853117539872062e+02, 0.,
               6.3950000000000000e+02, 0., 8.4853117539872062e+02,
               3.5950000000000000e+02, 0., 0., 1);

  std::cout << "K:\n" << K << std::endl;

  double prob = 0.999;
  double threshold = 1.0;

  std::vector<cv::Point2f> points1, points2;

  // Populate points1 and points2 with the matched keypoints
  for (const auto &match : goodMatches) {
    points1.push_back(keypoints1[match.queryIdx].pt);
    points2.push_back(keypoints2[match.trainIdx].pt);
  }

  cv::Mat E =
      cv::findEssentialMat(points1, points2, K, cv::RANSAC, prob, threshold);

  std::cout << "E:\n" << E << std::endl;

  // Normalize points using the intrinsic matrix K
  std::vector<cv::Point2f> normPoints1, normPoints2;
  cv::undistortPoints(points1, normPoints1, K, cv::noArray());
  cv::undistortPoints(points2, normPoints2, K, cv::noArray());

  double epipolarError = 0.0;
  for (size_t i = 0; i < points1.size(); ++i) {
    cv::Mat x1 =
        (cv::Mat_<double>(3, 1) << normPoints1[i].x, normPoints1[i].y, 1.0);
    cv::Mat x2 =
        (cv::Mat_<double>(3, 1) << normPoints2[i].x, normPoints2[i].y, 1.0);
    cv::Mat error = x2.t() * E * x1;
    // std::cout << "error:" << std::abs(error.at<double>(0, 0)) << std::endl;
    epipolarError += std::abs(error.at<double>(0, 0));
  }
  epipolarError /= points1.size();
  std::cout << "Average Epipolar Error: " << epipolarError << std::endl;

  // Decompose the essential matrix into R and t
  cv::Mat R, t, mask;
  cv::recoverPose(E, points1, points2, K, R, t, mask);

  // Compute the projection matrices
  cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F); // [I|0] for the first camera
  cv::Mat P2(3, 4, CV_64F);                // [R|t] for the second camera
  R.copyTo(P2(cv::Rect(0, 0, 3, 3))); // Copy R into the top-left 3x3 submatrix
  t.copyTo(P2(cv::Rect(3, 0, 1, 3))); // Copy t into the last column

  // Triangulate points
  cv::Mat points4D;
  cv::triangulatePoints(K * P1, K * P2, points1, points2, points4D);

  // Convert homogeneous coordinates to 3D points
  std::vector<cv::Point3f> points3D;

  // Prepare 3d points for logging
  std::vector<rerun::components::Position3D> point3d_positions;

  float clamp = 50.0f;
  float x, y, z;
  for (int i = 0; i < points4D.cols; ++i) {
    cv::Mat col = points4D.col(i);
    col /= col.at<float>(3); // Normalize by w

    x = col.at<float>(0);
    y = col.at<float>(1);
    z = col.at<float>(2);

    if ((std::abs(x) < clamp) && (std::abs(y) < clamp) &&
        (std::abs(z) < clamp)) {
      points3D.push_back(cv::Point3f(x, y, z));
      point3d_positions.push_back({x, y, z});
    }
  }

  // https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga1019495a2c8d1743ed5cc23fa0daff8c
  //  Output triangulated points
  for (const auto &pt : points3D) {
    std::cout << "3D Point: " << pt << std::endl;
  }

  const auto rec = rerun::RecordingStream("RIGHT_HAND_Y_DOWN");
  rec.spawn().exit_on_failure();

  rec.log_static(
      "world",
      rerun::ViewCoordinates::RIGHT_HAND_Y_DOWN); // OpenCV convention
  rec.log("world/xyz",
          rerun::Arrows3D::from_vectors(
              {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}})
              .with_colors({{255, 0, 0}, {0, 255, 0}, {0, 0, 255}}));

  rec.log("world/points", rerun::Points3D(point3d_positions));

  //   cv::KeyPoint::convert(keypoints1, points1);
  //   cv::KeyPoint::convert(keypoints2, points2);
  // cv::findFundamentalMat
  // cv::estimateTranslation3D
  cv::waitKey(0);
}

int main(int argc, char **argv) {

  //   std::string image_path1 =
  //   "../../images/correspondences_matching/000000.png"; std::string
  //   image_path2 = "../../images/correspondences_matching/000001.png";

  std::string image_path1 =
      "/home/behnam/workspace/slam-tutorials/images/0.png";
  std::string image_path2 =
      "/home/behnam/workspace/slam-tutorials/images/1.png";

  cv::Mat img1 = cv::imread(image_path1, cv::IMREAD_ANYCOLOR);
  cv::Mat img2 = cv::imread(image_path2, cv::IMREAD_ANYCOLOR);

  correspondencesMatching(img1, img2);

  return 0;
}
