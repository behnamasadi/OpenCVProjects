#include <opencv2/opencv.hpp>


void OPticalFlowLucasKanade()
{



/*
Obsolete functions:
CalcOpticalFlowHS -> computes the flow for every pixel of the first input image using the Horn and Schunck algorithm
CalcOpticalFlowLK -> computes the flow for every pixel of the first input image using the Lucas and Kanade algorithm .



To track sparse features, use calcOpticalFlowPyrLK().
To track all the pixels, use calcOpticalFlowFarneback().
calcOpticalFlowSF -> Calculate an optical flow using “SimpleFlow” algorithm.
*/



}



void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step,double, const cv::Scalar& color)
{
//    std::cout<<flow.channels() <<std::endl;
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
            cv::line(cflowmap, cv::Point(x,y), cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), color);
            cv::circle(cflowmap, cv::Point(x,y), 2, color, -1);
        }
}

void drawMotionToColor(const cv::Mat& flow,  cv::Mat &bgr)
{
    //extraxt x and y channels
    cv::Mat xy[2]; //X,Y
    cv::split(flow, xy);

    //calculate angle and magnitude
    cv::Mat magnitude, angle;
    cv::cartToPolar(xy[0], xy[1], magnitude, angle, true);

    //translate magnitude to range [0;1]
    double mag_max;
    cv::minMaxLoc(magnitude, 0, &mag_max);
    magnitude.convertTo(magnitude, -1, 1.0 / mag_max);

    //build hsv image
    cv::Mat _hsv[3], hsv;
    _hsv[0] = angle;
    _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magnitude;
    cv::merge(_hsv, 3, hsv);

    //convert to BGR and show
   /* cv::Mat bgr*/;//CV_32FC3 matrix
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
}

void OPticalFlowPyramidLucasKanade(cv::Mat &previous_image,cv::Mat &next_image, std::vector<cv::Point2f>& previous_image_points,
                           std::vector<cv::Point2f>& next_image_points,std::vector<uchar>& status)
{
    cv::Size winSize=cv::Size(21,21);
    std::vector<float> err;
    cv::TermCriteria termcrit=cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01);
    cv::calcOpticalFlowPyrLK(previous_image, next_image, previous_image_points, next_image_points, status, err, winSize, 3, termcrit, 0, 0.001);



    int indexCorrection = 0;
    for( int i=0; i<status.size(); i++)
       {  cv::Point2f pt = next_image_points.at(i- indexCorrection);
          if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))	{
                if((pt.x<0)||(pt.y<0))	{
                  status.at(i) = 0;
                }
                previous_image_points.erase (previous_image_points.begin() + (i - indexCorrection));
                next_image_points.erase (next_image_points.begin() + (i - indexCorrection));
                indexCorrection++;
          }

       }


}


void OpticalFlowFarneback(cv::Mat &previous_image,cv::Mat &next_image,cv::Mat &flow,double pyr_scale=0.5,
                                      int levels=5,int winsize=13,int numIters = 10,int poly_n=5,int poly_sigma=1.1, int flags=cv::OPTFLOW_FARNEBACK_GAUSSIAN)
{
/*
    pyr_scale=0.5; means a classical pyramid, where each next layer is twice smaller than the previous one.
    int levels=5;//number of pyramid layers including the initial image; levels=1 means that no extra layers are
    int winsize=13;//averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
    int numIters = 10; //number of iterations the algorithm does at each pyramid level.

    int poly_n=5;//size of the pixel neighborhood used to find polynomial expansion in each pixel; larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
    int poly_sigma=1.1;//standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
    flags=
    OPTFLOW_USE_INITIAL_FLOW-> uses the input flow as an initial flow approximation
    OPTFLOW_FARNEBACK_GAUSSIAN ->uses the Gaussian winsize Xwinsize filter instead of a box filter of the same size for optical flow estimation; usually, this option gives z more accurate flow than with a box filter, at the cost of lower speed; normally, winsize for a Gaussian window should be set to a larger value to achieve the same level of robustness.

*/
    cv::calcOpticalFlowFarneback(previous_image,next_image,flow,pyr_scale,levels,winsize,numIters,poly_n,poly_sigma,flags);
}

//using namespace cv;
//using namespace std;
//void temp()
//{
//    // Load two images and allocate other structures
//    Mat imgA = imread("../images/opticalflow/bt_0.png", CV_LOAD_IMAGE_GRAYSCALE);
//    Mat imgB = imread("../images/opticalflow/bt_1.png", CV_LOAD_IMAGE_GRAYSCALE);

//    Size img_sz = imgA.size();
//    Mat imgC(img_sz,1);

//    int win_size = 15;
//    int maxCorners = 20;
//    double qualityLevel = 0.05;
//    double minDistance = 5.0;
//    int blockSize = 3;
//    double k = 0.04;
//    std::vector<cv::Point2f> cornersA;
//    cornersA.reserve(maxCorners);
//    std::vector<cv::Point2f> cornersB;
////    cornersB.reserve(maxCorners);


//    goodFeaturesToTrack( imgA,cornersA,maxCorners,qualityLevel,minDistance,cv::Mat());
//    goodFeaturesToTrack( imgB,cornersB,maxCorners,qualityLevel,minDistance,cv::Mat());

//    cornerSubPix( imgA, cornersA, Size( win_size, win_size ), Size( -1, -1 ),
//                  TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03 ) );

////    cornerSubPix( imgB, cornersB, Size( win_size, win_size ), Size( -1, -1 ),
////                  TermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03 ) );

//    // Call Lucas Kanade algorithm

//    CvSize pyr_sz = Size( img_sz.width+8, img_sz.height/3 );

//    std::vector<uchar> features_found; //status : each element of the vector is set to 1 if the flow for the corresponding features has been found, otherwise, it is set to 0.
//    features_found.reserve(maxCorners);
//    std::vector<float> feature_errors;
//    feature_errors.reserve(maxCorners);

//    cornersB.clear();

//    calcOpticalFlowPyrLK( imgA, imgB, cornersA, cornersB, features_found, feature_errors ,
//        Size( win_size, win_size ), 5,
//         cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.3 ), 0 );

//    // Make an image of the results

//    for( int i=0; i < features_found.size(); i++ ){
////            cout<<"Error is "<<feature_errors[i]<<endl;
////            //continue;

////        cout<<"Got it"<<endl;
//        cout<< (int)features_found.at(i)<<endl;
//        Point p0( ceil( cornersA[i].x ), ceil( cornersA[i].y ) );
//        Point p1( ceil( cornersB[i].x ), ceil( cornersB[i].y ) );
//        line( imgC, p0, p1, CV_RGB(255,255,255), 2 );
//    }

//    namedWindow( "ImageA", 0 );
//    namedWindow( "ImageB", 0 );
//    namedWindow( "LKpyr_OpticalFlow", 0 );

//    imshow( "ImageA", imgA );
//    imshow( "ImageB", imgB );
//    imshow( "LKpyr_OpticalFlow", imgC );

//    cvWaitKey(0);

//    return ;
//}




void drawLukasOpticalFlow(cv::Mat &flow, std::vector<cv::Point2f> &corners_prev,std::vector<cv::Point2f> &corners_current,std::vector<uchar> &features_found)
{
//    std::cout<<"features_found.size():"<<features_found.size() <<std::endl;
    for( int i=0; i < features_found.size(); i++ )
    {
//        std::cout<< (int)features_found.at(i)<<std::endl;
        cv::Point p0( std::ceil( corners_prev[i].x ), std::ceil( corners_prev[i].y ) );
        cv::Point p1( std::ceil( corners_current[i].x ), std::ceil( corners_current[i].y ) );
        cv::line( flow, p0, p1, CV_RGB(255,0,0), 2 );
    }
}



static const double pi = 3.14159265358979323846;
inline static double square(int a)
{
return a * a;
}
 int number_of_features;
void draw(cv::Mat &frame1, std::vector<cv::Point2f> &frame1_features,std::vector<cv::Point2f> &frame2_features,std::vector<uchar> &optical_flow_found_feature)
{
    for(int i = 0; i < number_of_features; i++)
    {
        /* If Pyramidal Lucas Kanade didn't really find the feature, skip it. */
        if ( optical_flow_found_feature[i] == 0 ) continue;
        int line_thickness; line_thickness = 1;
        /* CV_RGB(red, green, blue) is the red, green, and blue components
        * of the color you want, each out of 255.
        */
        cv::Scalar line_color; line_color = CV_RGB(255,0,0);
        /* Let's make the flow field look nice with arrows. */
        /* The arrows will be a bit too short for a nice visualization because of the
       high framerate
        * (ie: there's not much motion between the frames). So let's lengthen them
       by a factor of 3.
        */
        cv::Point p,q;
        p.x = (int) frame1_features[i].x;
        p.y = (int) frame1_features[i].y;
        q.x = (int) frame2_features[i].x;
        q.y = (int) frame2_features[i].y;
        double angle; angle = atan2( (double) p.y - q.y, (double) p.x - q.x );
        double hypotenuse;
        hypotenuse = sqrt( square(p.y - q.y) + square(p.x - q.x) )       ;
        /* Here we lengthen the arrow by a factor of three. */
        q.x = (int) (p.x - 3 * hypotenuse * cos(angle));
        q.y = (int) (p.y - 3 * hypotenuse * sin(angle));
        /* Now we draw the main line of the arrow. */
       /* "frame1" is the frame to draw on.
        * "p" is the point where the line begins.
        * "q" is the point where the line stops.
        * "CV_AA" means antialiased drawing.
        * "0" means no fractional bits in the center cooridinate or radius.
        */
        cv::line( frame1, p, q, line_color, 1 );
        //cv::line(img, cv::Point(100,100), cv::Point(200,200), cv::Scalar(0,255,0), 1);

        /* Now draw the tips of the arrow. I do some scaling so that the
        * tips look proportional to the main line of the arrow.
        */
        p.x = (int) (q.x + 9 * cos(angle + pi / 4));
        p.y = (int) (q.y + 9 * sin(angle + pi / 4));
        cv::line( frame1, p, q, line_color,  0 );
        p.x = (int) (q.x + 9 * cos(angle - pi / 4));
        p.y = (int) (q.y + 9 * sin(angle - pi / 4));
        cv::line( frame1, p, q, line_color,  0 );
    }

}


//void OpticalFlowPyramidLukas(cv::Mat &current_gray, cv::Mat &prev_gray,cv::Mat &LKpyr_OpticalFlow)
//{

//}

void OpticalFlowPyramidLukas_test(int argc, char** argv)
{
/*
    std::string camera_calibration_path="../data/front_webcam.yml";
    cv::FileStorage fs(camera_calibration_path,cv::FileStorage::READ);
    cv::Mat camera_matrix, distortion_coefficient;
    fs["camera_matrix"]>>camera_matrix;
    fs["distortion_coefficients"]>>distortion_coefficient;

    std::cout<<"Camera Matrix:" <<std::endl;
    std::cout<<camera_matrix <<std::endl;
    std::cout<<"Fx: " <<camera_matrix.at<double>(0,0) <<std::endl;
    std::cout<<"Fy: " <<camera_matrix.at<double>(1,1) <<std::endl;

    std::cout<<"Cx: " <<camera_matrix.at<double>(0,2) <<std::endl;
    std::cout<<"Cy: " <<camera_matrix.at<double>(1,2) <<std::endl;
    std::cout<< "Distortion Coefficient:"<<std::endl;
    std::cout<<distortion_coefficient <<std::endl;

    std::cout<<"K1: "<<distortion_coefficient.at<double>(0,0) <<std::endl;
    std::cout<<"K2: "<<distortion_coefficient.at<double>(0,1) <<std::endl;
    std::cout<<"P1: "<<distortion_coefficient.at<double>(0,2) <<std::endl;
    std::cout<<"P2: "<<distortion_coefficient.at<double>(0,3) <<std::endl;
    std::cout<<"K3: "<<distortion_coefficient.at<double>(0,4) <<std::endl;

*/


    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);

    cv::Mat E, R, t;


    int win_size = 15;
    int maxCorners = 100;
    number_of_features=maxCorners;
    double qualityLevel = 0.05;
    double minDistance = 5.0;
    int blockSize = 3;
    double k = 0.04;
    std::vector<cv::Point2f> corners_prev;
    std::vector<cv::Point2f> corners_current;
//    cornersB.reserve(maxCorners);


    std::vector<uchar> features_found; //status : each element of the vector is set to 1 if the flow for the corresponding features has been found, otherwise, it is set to 0.
    features_found.reserve(maxCorners);
    std::vector<float> feature_errors;
    feature_errors.reserve(maxCorners);




    cv::Mat current_gray, previous_gray,frame;


//    cv::namedWindow( "previous_gray", 0 );
//    cv::namedWindow( "current_gray", 0 );
    cv::namedWindow( "LKpyr_OpticalFlow", 0 );

    double x,y;
    double x_ref, y_ref;
    x_ref=0;
    y_ref=0;

    x=x_ref;
    y=y_ref;

    cv::VideoCapture vid;
    if(argc>1)
    {
        vid.open(argv[1]);
    }else
    {
        cv::VideoCapture camera(0);
        vid=camera;
    }

    for(;;)
    {
        vid >> frame;
        cv::cvtColor(frame, current_gray, cv::COLOR_BGR2GRAY);

        if( !previous_gray.empty() )
        {
            cv::Size img_sz = previous_gray.size();

//            cv::Mat LKpyr_OpticalFlow=cv::Mat::zeros(img_sz,1);
            //cv::Mat LKpyr_OpticalFlow=previous_gray.clone();
            cv::Mat LKpyr_OpticalFlow=frame.clone();

            cv::goodFeaturesToTrack( previous_gray,corners_prev,maxCorners,qualityLevel,minDistance,cv::Mat());
            cv::goodFeaturesToTrack( current_gray,corners_current,maxCorners,qualityLevel,minDistance,cv::Mat());
            cv::cornerSubPix( previous_gray, corners_prev, cv::Size( win_size, win_size ), cv::Size( -1, -1 ),cv::TermCriteria( cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS, 20, 0.03 ) );
            cv::calcOpticalFlowPyrLK( previous_gray, current_gray, corners_prev, corners_current, features_found, feature_errors , cv::Size( win_size, win_size ), 5, cv::TermCriteria( cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS, 20, 0.3 ), 0 );


//            int indexCorrection = 0;
//            for( int i=0; i<features_found.size(); i++)
//            {
//                cv::Point2f pt = corners_current.at(i- indexCorrection);
//                if ((features_found.at(i) == 0)||(pt.x<0)||(pt.y<0))
//                {
//                    if((pt.x<0)||(pt.y<0))
//                    {
//                        features_found.at(i) = 0;
//                    }
//                    corners_prev.erase (corners_prev.begin() + (i - indexCorrection));
//                    corners_current.erase (corners_current.begin() + (i - indexCorrection));
//                    indexCorrection++;
//                }

//            }


            E=cv::findEssentialMat(corners_current,corners_prev,camera_matrix,cv::RANSAC);
//            std::cout <<"Essential: " <<E <<std::endl;
            cv::recoverPose(E, corners_current, corners_prev,camera_matrix, R, t);






//            std::cout <<"Trasnlation" <<t <<std::endl;
//            std::cout <<"X: " <<t.at<double>(0,0) <<std::endl;
//            std::cout <<"Y: " <<t.at<double>(1,0) <<std::endl;
/*
            std::cout <<x << ","<<y <<std::endl;

            x=x+t.at<double>(0,0);
            y=y+t.at<double>(1,0);
*/

//            std::cout <<t.at<double>(0,0) << ","<<t.at<double>(1,0) <<std::endl;



//            drawLukasOpticalFlow(LKpyr_OpticalFlow,corners_prev,corners_prev,features_found);

            draw(LKpyr_OpticalFlow, corners_prev,corners_prev,features_found);
//            cv::imshow( "previous_gray", previous_gray );
//            cv::imshow( "current_gray", current_gray );
            cv::imshow( "LKpyr_OpticalFlow", LKpyr_OpticalFlow );
        }
        if(cv::waitKey(30)>=0)
            break;
        std::swap(previous_gray, current_gray);
    }


}
void OpticalFlowFarneback_test(int argc, char** argv)
{


    cv::VideoCapture vid;
    if(argc>1)
    {
        vid.open(argv[1]);
    }else
    {
        cv::VideoCapture camera(0);
        vid=camera;
    }


    cv::Mat flow, cflow,bgr ,frame;
    cv::Mat gray, prevgray, uflow;
    cv::namedWindow("flow", 1);
    cv::namedWindow("motiontoflow", 1);


    double pyr_scale=0.5;
    int levels=5;
    int winsize=13;
    int numIters = 10;
    int poly_n=5;
    int poly_sigma=1.1;
    int flags=cv::OPTFLOW_FARNEBACK_GAUSSIAN;


    for(;;)
    {
        vid >> frame;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        if( !prevgray.empty() )
        {
            OpticalFlowFarneback(prevgray,gray,uflow,pyr_scale,levels,winsize,numIters,poly_n,poly_sigma,flags);
            cv::cvtColor(prevgray, cflow, cv::COLOR_GRAY2BGR);
            uflow.copyTo(flow);
            drawOptFlowMap(flow, cflow, 16, 1.5, cv::Scalar(0, 255, 0));
            drawMotionToColor(flow,  bgr);
            cv::imshow("flow", cflow);
            cv::imshow("motiontoflow", bgr);
        }
        if(cv::waitKey(30)>=0)
            break;
        std::swap(prevgray, gray);
    }
    return ;
}

std::string ZeroPadNumber(int num)
{
    std::stringstream ss;

    // the number is converted to string with the help of stringstream
    ss << num;
    std::string ret;
    ss >> ret;

    // Append zero chars
    int str_length = ret.length();
    for (int i = 0; i < 6 - str_length; i++)
        ret = "0" + ret;
    return ret;
}


void temp2(int argc, char ** argv)
{
    cv::VideoCapture vid(argv[1]);



    cv::Mat frame;
    std::string file_name,file_path;
    file_path="frames/";


    std::cout<< int(vid.get(cv::CAP_PROP_FRAME_COUNT)) <<std::endl;

    int totoal_number_of_frame=int(vid.get(cv::CAP_PROP_FRAME_COUNT));
    for(std::size_t i=0;i<totoal_number_of_frame;i=i+1)
    {
        vid>>frame;
        file_name=ZeroPadNumber(i)+".png";
        std::cout<< file_name <<std::endl;
        cv::imwrite(file_path+file_name,frame);
//        i++;
    }


}

int main(int argc, char** argv)
{

//    OpticalFlowFarneback_test(argc, argv);
    OpticalFlowPyramidLukas_test(argc, argv);
//    temp2(argc, argv);
}


