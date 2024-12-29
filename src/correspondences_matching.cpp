#include "arguments_parser.hpp"
#include <opencv2/opencv.hpp>

void correspondencesMatching(cv::Mat img1, cv::Mat img2) {

  // Create SIFT (floating-point descriptors)
  cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;

  sift->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
  sift->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);

  //////////////////////// BFMatcher ////////////////////////////

  cv::BFMatcher matcher(cv::NORM_L2);
  std::vector<cv::DMatch> matches;
  cv::Mat img_matches;
  matcher.match(descriptors1, descriptors2, matches);
  cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);

  cv::imshow("BFMatcher", img_matches);
  cv::imwrite("BFMatcher.png", img_matches);
  cv::waitKey(0);

  //////////////////////// knnMatches ////////////////////////////

  std::vector<std::vector<cv::DMatch>> knnMatches;
  int k = 5;
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
  cv::imwrite("KNN_Matches.png", img_knn_matches);

  cv::waitKey(0);

  //////////////////////// radius matching ////////////////////////////
  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  orb->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
  orb->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

  // Use BFMatcher with NORM_HAMMING (suitable for ORB descriptors)
  cv::BFMatcher radius_matcher(cv::NORM_HAMMING);

  // Perform radius matching
  const float maxDistance = 100.0f; // Radius threshold
  std::vector<std::vector<cv::DMatch>> radiusMatches;
  matcher.radiusMatch(descriptors1, descriptors2, radiusMatches, maxDistance);

  // Filter and collect matches for visualization
  std::vector<cv::DMatch> radiusGoodMatches;
  for (const auto &matches : radiusMatches) {
    for (const auto &match : matches) {
      if (match.distance < maxDistance) {
        radiusGoodMatches.push_back(match);
      }
    }
  }

  // Draw matches
  cv::Mat imgMatches;
  cv::drawMatches(img1, keypoints1, img2, keypoints2, radiusMatches,
                  imgMatches);

  // Display the result
  cv::imshow("Radius Match", imgMatches);
  cv::imwrite("Radius_Match.png", img_knn_matches);

  cv::waitKey(0);
}

int main(int argc, char **argv) {

  std::string image_path1 = "../../images/correspondences_matching/000000.png";

  std::string image_path2 = "../../images/correspondences_matching/000001.png";

  cv::Mat img1 = cv::imread(image_path1, cv::IMREAD_ANYCOLOR);
  cv::Mat img2 = cv::imread(image_path2, cv::IMREAD_ANYCOLOR);

  correspondencesMatching(img1, img2);

  return 0;
}