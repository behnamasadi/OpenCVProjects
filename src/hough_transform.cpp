//https://en.wikipedia.org/wiki/Hough_transform


//void HoughLineTransform(Mat src )
//{
////	src  is in grayscale
//	Mat dst, coloerfull_dst;
//	Canny(src, dst, 50, 200, 3);
////	we turn our dst matrix to colorfull so we can draw red lines
//	cvtColor(dst, coloerfull_dst, CV_GRAY2BGR);

//	#if 0
//		// this vector contains rho and theta
//		vector<Vec2f> lines;

////		dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
////		lines: A vector that will store the parameters  of the detected lines
////		rho : The resolution of the parameter  in pixels. We use 1 pixel.
////		theta: The resolution of the parameter  in radians. We use 1 degree (CV_PI/180)
////		threshold: The minimum number of intersections to “detect” a line
////		srn and stn: Default parameters to zero. Check OpenCV reference for more info.

//		double rho_resolution_parameter_in_pixels=1;
//		double theta_resolution_in_radian=CV_PI/180;
//		int threshold=100;
//		double srn=0;
//		double stn=0;
//		HoughLines(dst, lines, rho_resolution_parameter_in_pixels, theta_resolution_in_radian, 100, 0, 0 );
//		for( size_t i = 0; i < lines.size(); i++ )
//		{
//			 float rho = lines[i][0], theta = lines[i][1];
//			 Point pt1, pt2;
//			 double a = cos(theta), b = sin(theta);
//			 double x0 = a*rho, y0 = b*rho;
//			 pt1.x = cvRound(x0 + 1000*(-b));
//			 pt1.y = cvRound(y0 + 1000*(a));
//			 pt2.x = cvRound(x0 - 1000*(-b));
//			 pt2.y = cvRound(y0 - 1000*(a));
//			 line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
//		}
//	#else
//		vector<Vec4i> lines;
//		/*
//		dst: Output of the edge detector. It should be a grayscale image (although in fact it is a binary one)
//		lines: A vector that will store the parameters  of the detected lines
//		rho : The resolution of the parameter  in pixels. We use 1 pixel.
//		theta: The resolution of the parameter  in radians. We use 1 degree (CV_PI/180)
//		threshold: The minimum number of intersections to “detect” a line
//		minLinLength: The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
//		maxLineGap: The maximum gap between two points to be considered in the same line.
//		*/
//		double rho_resolution_parameter_in_pixels=1;
//		double theta_resolution_in_radian=CV_PI/180;
//		int minimum_number_of_intersections_to_detect_line=50;
//		double minLinLength=50;
//		double maximum_gap_between_two_points_to_be_considered_in_the_same_line=10;

//		HoughLinesP(dst, lines, rho_resolution_parameter_in_pixels, theta_resolution_in_radian,
//				minimum_number_of_intersections_to_detect_line, minLinLength, maximum_gap_between_two_points_to_be_considered_in_the_same_line);
//		for( size_t i = 0; i < lines.size(); i++ )
//		{
//			Vec4i l = lines[i];
//			line( coloerfull_dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
//		}
//	#endif
//	imshow("source", src);
//	imshow("detected lines", coloerfull_dst);

//	waitKey();


//}

//void HoughLineTransform_Test(char ** argv)
//{
//	Mat src = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE );
//	HoughLineTransform(src );
//}

//void HoughTransformCircle(IplImage *img_src)
//{
//	Mat src(img_src,false);
//	Mat src_gray;

////	Convert it to gray
//	cvtColor( src, src_gray, cv::COLOR_BGR2GRAY );

////	Convert it to gray
//	cvtColor( src, src_gray, cv::COLOR_BGR2GRAY );

////	Reduce the noise so we avoid false circle detection
//	GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );

//	vector<Vec3f> circles;
//	int UpperthresholdforCannyEdgeDetector=10;
////	Apply the Hough Transform to find the circles
////	src_gray: Input image (grayscale)
////	circles: A vector that stores sets of 3 values:  for each detected circle.
////	CV_HOUGH_GRADIENT: Define the detection method. Currently this is the only one available in OpenCV
////	dp = 1: The inverse ratio of resolution
////	min_dist = src_gray.rows/8: Minimum distance between detected centers
////	param_1 = 200: Upper threshold for the internal Canny edge detector
////	param_2 = 100*: Threshold for center detection.
////	min_radius = 0: Minimum radio to be detected. If unknown, put zero as default.
////	max_radius = 0: Maximum radius to be detected. If unknown, put zero as default
//	HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/8, UpperthresholdforCannyEdgeDetector, 100, 0, 0 );

//	/// Draw the circles detected
//	for( size_t i = 0; i < circles.size(); i++ )
//	{
//		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
//		int radius = cvRound(circles[i][2]);
//		// circle center
//		circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
//		// circle outline
//		circle( src, center, radius, Scalar(0,0,255), 3, 8, 0 );
//	}

////	Show your results
//	namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
//	imshow( "Hough Circle Transform Demo", src );
//	waitKey(0);
//	return ;
//}

//void HoughTransformCircle_Test(char ** argv)
//{
//	IplImage *img_src= cvLoadImage(argv[1]);
//	HoughTransformCircle( img_src);


//}


void houghLineTransform()
{

}

void houghTransformCircle()
{

}

int main()
{
    return 0;
}