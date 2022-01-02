#include <opencv4/opencv2/opencv.hpp>

/*
Edge detection:

Canny
Deriche
Differential
Sobel
PrewittRoberts 
cross
*/

//Canny is based on first derivative

/// Global variables

cv::Mat canny_edge_src, canny_edge_src_gray;
cv:: Mat canny_edge_dst, canny_edge_detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";


//void CannyEdgeDetector(int, void*)
//{
//    /*
//    Step 1:
//        Apply a Gaussian blur
//    Step 2:
//        Find edge gradient strength and angle
//    Step 3:
//        classify direction of gradient into 0,1,2,3
//        round the gradient direction theta to nearest 45 degree
//        i.e if gradient angle is 23 degree it will fall into 0 interval

//    Step 4:
//        Suppress non-maximum edges
//        for a 3x3 neighbor of each pixel
//        select two neigbors which are in the direction of gradient, i.e if the direction of gradient at point x is 1, we select the neighbor at top right and
//        bottom left:

//        3 2 1
//        0 x 0
//        1 2 3

//        and if the magnitude of gradient is biger than both keep it otherwise set it to zero

//    Step 5:
//        Hysteresis thresholding: use of two thresholds, a high   HT and a low LT:
//        pixel value in non-maxima suppressed image M (x,y) greater than HT is immediately accepted as an edge pixel
//        pixel value in non-maxima suppressed image M (x,y) below the LT is immediately rejected.
//        pixels in M (x,y) whose values lie between the two thresholds are accepted if they are connected to the already detected edge pixels
//    */

//    /// Reduce noise with a kernel 3x3
//    blur( canny_edge_src_gray, canny_edge_detected_edges, Size(3,3) );

//    /// Canny detector
//    cv::Canny( canny_edge_detected_edges, canny_edge_detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

//    /// Using Canny's output as a mask, we display our result
//    canny_edge_dst = Scalar::all(0);

//    canny_edge_src.copyTo( canny_edge_dst, canny_edge_detected_edges);
//    imshow( window_name, canny_edge_dst );
//}

//int CannyEdgeDetector_Test( char** argv )
//{
//    /// Load an image
//    canny_edge_src = imread( argv[1] );

//    if( !canny_edge_src.data )
//    { return -1; }

//    /// Create a matrix of the same type and size as src (for dst)
//    canny_edge_dst.create( canny_edge_src.size(), canny_edge_src.type() );

//    /// Convert the image to grayscale
//    cvtColor( canny_edge_src, canny_edge_src_gray, cv::COLOR_BGR2GRAY );

//    /// Create a window
//    namedWindow( window_name, CV_WINDOW_AUTOSIZE );

//    /// Create a Trackbar for user to enter threshold
//    createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyEdgeDetector );

//    /// Show the image
//    CannyEdgeDetector(0, 0);

//    /// Wait until user exit program by pressing a key
//    waitKey(0);

//    return 0;
//}


//Sobel is based on first derivative so we have to set a threshold value after applying the algorithm to threshold the edges
//void SobelEdgeDetector(char ** argv)
//{
//    Mat src, src_gray;
//    Mat grad;
//    char* window_name = "Sobel Demo - Simple Edge Detector";
//    int scale = 1;
//    int delta = 0;
////    If src is 8-bit then the dst must be of depth IPL_DEPTH_16S to avoid overflow.
//    int ddepth = CV_16S;

//    int c;

//    /// Load an image
//    src = imread( argv[1] );

//    if( !src.data )
//    { return ; }

//    GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );

//    /// Convert it to gray
//    cvtColor( src, src_gray, CV_RGB2GRAY );

//    /// Create window
//    namedWindow( window_name, CV_WINDOW_AUTOSIZE );

//    /// Generate grad_x and grad_y
//    Mat grad_x, grad_y;
//    Mat abs_grad_x, abs_grad_y;
//    int dx_order;
//    int dy_order;
///*
//    kernel_size=aperture_size
//    The aperture_size parameter should be odd and is the width (and the height) of the square fi lter. Currently, aperture_sizes of 1, 3, 5, and 7 are supported.
//    The larger kernels give a better approximation to the derivative because the smaller kernels are very sensitive to noise.
//*/
//    int kernel_size=3;
   
    

//    /// Gradient X
//    //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
//    dx_order=1;
//    dy_order=0;
//    Sobel( src_gray, grad_x, ddepth, dx_order, dy_order, kernel_size, scale, delta, BORDER_DEFAULT );
//    //because we can't display negative values so we have to convert the scale
//    convertScaleAbs( grad_x, abs_grad_x );

//    /// Gradient Y
//    dx_order=0;
//    dy_order=1;
//    //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
//    Sobel( src_gray, grad_y, ddepth, dx_order, dy_order, kernel_size, scale, delta, BORDER_DEFAULT );
//    //because we can't display negative values so we have to convert the scale
//    convertScaleAbs( grad_y, abs_grad_y );

//    /// Total Gradient (approximate)
//    //TotalGradient=sqrt( grad_y^2 + grad_t^2 ) ==>approximate  =0.5*abs_grad_x+ 0.5*abs_grad_y
//    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

//    imshow( window_name, grad );

//    waitKey(0);
//}

int main()
{
    return 0;
}