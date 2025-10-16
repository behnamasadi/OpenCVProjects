#include <opencv2/opencv.hpp>

template <typename T> void printArray(std::vector<T> array) {
  for (auto element : array)
    std::cout << element << std::endl;
}

void getHomographyMatrix() {
  std::vector<cv::Point2f> point_on_plane1;
  std::vector<cv::Point2f> point_on_plane2;
  std::vector<cv::Point2f> obj_projection;

  cv::Point2f A, B, C, D, A_P, B_P, C_P, D_P;
  A.x = 0;
  A.y = 0;

  B.x = 150;
  B.y = 0;

  C.x = 150;
  C.y = 150;

  D.x = 0;
  D.y = 150;

  point_on_plane1.push_back(A);
  point_on_plane1.push_back(B);
  point_on_plane1.push_back(C);
  point_on_plane1.push_back(D);

  A_P.x = 100;
  A_P.y = 100;

  B_P.x = 200;
  B_P.y = 80;

  C_P.x = 220;
  C_P.y = 80;

  D_P.x = 100;
  D_P.y = 200;

  point_on_plane2.push_back(A_P);
  point_on_plane2.push_back(B_P);
  point_on_plane2.push_back(C_P);
  point_on_plane2.push_back(D_P);

  std::cout << "Points in plane 1" << std::endl;
  printArray(point_on_plane1);

  std::cout << "Points in plane 2" << std::endl;
  printArray(point_on_plane2);

  cv::Mat homographyMatrix =
      cv::getPerspectiveTransform(point_on_plane1, point_on_plane2);
  std::cout << "Estimated Homography Matrix using getPerspectiveTransform:"
            << std::endl;
  std::cout << homographyMatrix << std::endl;

  cv::Mat H = cv::findHomography(point_on_plane1, point_on_plane2, 0);

  std::cout << "Estimated Homography Matrix using findHomography:" << std::endl;

  std::cout << H << std::endl;

  std::cout
      << "Projecting points in plane 1 with our estimated Homography Matrix is:"
      << std::endl;

  cv::perspectiveTransform(point_on_plane1, obj_projection, homographyMatrix);
  for (std::size_t i = 0; i < obj_projection.size(); i++) {
    std::cout << obj_projection.at(i).x << "," << obj_projection.at(i).y
              << std::endl;
  }
}

int main(int argc, char **argv) { getHomographyMatrix(); }
