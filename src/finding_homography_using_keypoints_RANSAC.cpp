#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

void findHomographyUsingKeypointsRANSAC(int argc, char **argv) {
  ///*
  // The input arrays src_points and dst_points can be either
  // 1)N-by-2(pixel coordinates) matrices
  // 2)N-by-3 (homogeneous coordinates)	matrices
  // The final argument, homography, is just a 	3-by-3 matrix to
  //*/

  //    if( argc != 3 )
  //    {
  //        printf(" Usage: ./homography <img1> <img2>\n");
  //        return;
  //    }

  //    cv::Mat img_object = cv::imread( argv[1], cv::IMREAD_GRAYSCALE );
  //    cv::Mat img_scene = cv::imread( argv[2], cv::IMREAD_GRAYSCALE);

  //    if( !img_object.data || !img_scene.data )
  //    {
  //        printf(" --(!) Error reading images \n"); return ;
  //    }

  //    cv::Mat img_object_gray,img_scene_gray;

  //    cv::cvtColor(img_object,img_object_gray,cv::COLOR_BGR2GRAY);
  //    cv::cvtColor(img_scene,img_scene_gray,cv::COLOR_BGR2GRAY);

  //    //-- Step 1: Detect the keypoints using SURF Detector
  ////    int minHessian = 400;
  ////    cv::SiftFeatureDetector detector( minHessian );
  ////    cv::SiftFeatureDetector detector;

  //    std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;

  ////    detector.detect( img_object, keypoints_object );
  ////    detector.detect( img_scene, keypoints_scene );

  //    cv::RNG rng(12345);
  //    std::vector<cv::Point2f> img_object_corners,img_scene_corners;
  //    double qualityLevel = 0.01;
  //    double minDistance = 10;
  //    int blockSize = 3;
  //    bool useHarrisDetector = false;
  //    double k = 0.04;
  //        int maxCorners = 100;

  //    cv::goodFeaturesToTrack( img_scene_gray,
  //                         img_scene_corners,
  //                         maxCorners,
  //                         qualityLevel,
  //                         minDistance,
  //                         cv::Mat(),
  //                         blockSize,
  //                         useHarrisDetector,
  //                         k );

  //    cv::goodFeaturesToTrack( img_object_gray,
  //                         img_object_corners,
  //                         maxCorners,
  //                         qualityLevel,
  //                         minDistance,
  //                         cv::Mat(),
  //                         blockSize,
  //                         useHarrisDetector,
  //                         k );

  //    //-- Step 2: Calculate descriptors (feature vectors)
  //    //cv::SurfDescriptorExtractor extractor;
  //    cv::SiftFeatureDetector extractor;

  //    cv::Mat descriptors_object, descriptors_scene;

  //    extractor.compute( img_object, keypoints_object, descriptors_object );
  //    extractor.compute( img_scene, keypoints_scene, descriptors_scene );

  //    //-- Step 3: Matching descriptor vectors using FLANN matcher
  //    cv::FlannBasedMatcher matcher;
  //    std::vector< cv::DMatch > matches;
  //    matcher.match( descriptors_object, descriptors_scene, matches );

  //    double max_dist = 0; double min_dist = 100;

  //    //-- Quick calculation of max and min distances between keypoints
  //    for( int i = 0; i < descriptors_object.rows; i++ )
  //    {   double dist = matches[i].distance;
  //        if( dist < min_dist ) min_dist = dist;
  //        if( dist > max_dist ) max_dist = dist;
  //    }

  //    printf("-- Max dist : %f \n", max_dist );
  //    printf("-- Min dist : %f \n", min_dist );

  //    //-- Draw only "good" matches (i.e. whose distance is less than
  //    3*min_dist ) std::vector< cv::DMatch > good_matches;

  //    for( int i = 0; i < descriptors_object.rows; i++ )
  //    { if( matches[i].distance < 1.5*min_dist )
  //        {
  //            good_matches.push_back( matches[i]);
  //        }
  //    }

  //    cv::Mat img_matches;
  //    cv::drawMatches( img_object, keypoints_object, img_scene,
  //    keypoints_scene, good_matches, img_matches, cv::Scalar::all(-1),
  //    cv::Scalar::all(-1), std::vector<char>(),
  //    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  //    //-- Localize the object from img_1 in img_2
  //    std::vector<cv::Point2f> obj;
  //    std::vector<cv::Point2f> scene;

  //    for( size_t i = 0; i < good_matches.size(); i++ )
  //    {
  //    //-- Get the keypoints from the good matches
  //        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
  //        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  //    }

  //    cv::Mat H = findHomography( obj, scene, cv::RANSAC );

  //    //-- Get the corners from the image_1 ( the object to be "detected" )
  //    std::vector<cv::Point2f> obj_corners(4);
  //    obj_corners[0] = cv::Point(0,0); obj_corners[1] = cv::Point(
  //    img_object.cols, 0 ); obj_corners[2] = cv::Point( img_object.cols,
  //    img_object.rows ); obj_corners[3] = cv::Point( 0, img_object.rows );
  //    std::vector<cv::Point2f> scene_corners(4);

  //    cv::perspectiveTransform( obj_corners, scene_corners, H);

  //    //-- Draw lines between the corners (the mapped object in the scene -
  //    image_2 ) cv::Point2f offset( (float)img_object.cols, 0); cv::line(
  //    img_matches, scene_corners[0] + offset, scene_corners[1] + offset,
  //    cv::Scalar(0, 255, 0), 4 ); cv::line( img_matches, scene_corners[1] +
  //    offset, scene_corners[2] + offset, cv::Scalar( 0, 255, 0), 4 );
  //    cv::line( img_matches, scene_corners[2] + offset, scene_corners[3] +
  //    offset, cv::Scalar( 0, 255, 0), 4 ); cv::line( img_matches,
  //    scene_corners[3] + offset, scene_corners[0] + offset, cv::Scalar( 0,
  //    255, 0), 4 );

  //    //-- Show detected matches
  //    cv::imshow( "Good Matches & Object detection", img_matches );

  //    cv::waitKey(0);

  //    return ;
}

int main(int argc, char **argv) { findHomographyUsingKeypointsRANSAC(argc,argv); }
