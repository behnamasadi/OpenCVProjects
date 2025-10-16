#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/types.hpp>

double computeReprojectionErrors( const std::vector<std::vector<cv::Point3f> >& objectPoints,
                          const std::vector<std::vector<cv::Point2f> >& imagePoints,
                          const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
                          const cv::Mat& cameraMatrix , const cv::Mat& distCoeffs,
                          std::vector<float>& perViewErrors)
{
    std::vector<cv::Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for( i = 0; i < (int)objectPoints.size(); ++i )
    {
      cv::projectPoints( cv::Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,  // project
                                           distCoeffs, imagePoints2);
      err = norm(cv::Mat(imagePoints[i]), cv::Mat(imagePoints2),  cv::NORM_L2 );              // difference

      int n = (int)objectPoints[i].size();
      perViewErrors[i] = (float) std::sqrt(err*err/n);                        // save for this view
      totalErr        += err*err;                                             // sum it up
      totalPoints     += n;
    }

    return std::sqrt(totalErr/totalPoints);              // calculate the arithmetical mean
}


void camera_calibration_example(cv::Mat &camera_matrix, cv::Mat &distortion_coefficients, std::vector<cv::Mat> &rvecs, std::vector<cv::Mat> &tvecs)
{
    int numBoards;
    int numCornersHor;
    int numCornersVer;
    float square_size;
//    char file_name[256];


    numBoards = 10;
    numCornersHor=9;
    numCornersVer=6;
    square_size= 0.025192;
    std::string file_name="fron_webcam";

/*
    printf("Enter number of corners along width: ");
    scanf("%d", &numCornersHor);

    printf("Enter number of corners along height: ");
    scanf("%d", &numCornersVer);

    printf("Enter number of boards(number of chess boards desired for calibration, minimum 4): ");
    scanf("%d", &numBoards);

    printf("square size im meter: ");
    scanf("%f", &square_size);

    printf("file name to save calibration matrix (YAML file): ");
    scanf("%s", &file_name);
*/
    printf("Press n to acquire next image");


    int numSquares = numCornersHor * numCornersVer;
    cv::Size board_sz = cv::Size(numCornersHor, numCornersVer);

    //we set it VideoCapture(1)
    cv::VideoCapture capture = cv::VideoCapture(0);
    std::vector<std::vector<cv::Point3f> > object_points;
    std::vector<std::vector<cv::Point2f> > image_points;
    std::vector<cv::Point2f> corners;
    int successes=0;
    cv::Mat image;
    cv::Mat gray_image;
    capture >> image;
    std::vector<cv::Point3f> obj;
    for(int j=0;j<numSquares;j++)
    {
        obj.push_back(cv::Point3f(  (j%numCornersHor) *square_size,(j/numCornersHor *square_size ) ,0.0f));
//        std::cout<<(j%numCornersHor) *square_size<<std::endl;
//        std::cout<<(j/numCornersHor *square_size )<<std::endl;
//        std::cout<<"--------------------------------------"<<std::endl;
    }


    while(successes<numBoards)
    {
        cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
        bool found = findChessboardCorners(image, board_sz, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FILTER_QUADS);
        if(found)
        {
            cv::cornerSubPix(gray_image, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS, 30, 0.1));
            //drawChessboardCorners(gray_image, board_sz, corners, found);
            drawChessboardCorners(image, board_sz, corners, found);
        }
        imshow("win1", image);
//        imshow("win2", gray_image);

        capture >> image;

        int key=cv::waitKey(1);


        if((char)key==(char)27)
            return ;

        if((char)key==(char)110 && found!=0)
        {
            std::cout<<"Snap stored!"<<std::endl;

            image_points.push_back(corners);
            object_points.push_back(obj);
            successes++;
            if(successes>=numBoards)
                break;
        }
    }

    //http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    cv::calibrateCamera(object_points, image_points, image.size(), camera_matrix, distortion_coefficients, rvecs, tvecs);

    std::cout<<"Trasnlations & Rotations" <<std::endl;
    for(std::size_t i=0;i<tvecs.size();i++)
    {
        std::cout<<"Trasnlation" <<std::endl;
        std::cout<<tvecs.at(i)<<std::endl;
        std::cout<<"Rotations" <<std::endl;
        std::cout<<rvecs.at(i)<<std::endl;

    }

    std::vector<float> per_view_errors;
    computeReprojectionErrors(object_points,image_points, rvecs, tvecs,  camera_matrix , distortion_coefficients,per_view_errors);

    std::cout<<"errors:" <<std::endl;
    for(std::size_t i=0;i<per_view_errors.size();i++)
    {
        std::cout<<"projection error:" <<per_view_errors.at(i)<<std::endl;

    }




    cv::FileStorage fs( std::string(file_name), cv::FileStorage::WRITE );


    fs << "camera_matrix"  <<camera_matrix;
    fs << "distortion_coefficients" << distortion_coefficients;

    if( !rvecs.empty() && !tvecs.empty() )
    {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        cv::Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());
        for( int i = 0; i < (int)rvecs.size(); i++ )
        {
            cv::Mat r = bigmat(cv::Range(i, i+1), cv::Range(0,3));
            cv::Mat t = bigmat(cv::Range(i, i+1), cv::Range(3,6));

            CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
            CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
            //*.t() is MatExpr (not Mat) so we can use assignment operator
            r = rvecs[i].t();
            t = tvecs[i].t();
        }
        //cv::FileStorage::writeComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        fs << "extrinsic_parameters" << bigmat;
    }
}




void undistort_example(int argc, char** argv)
{
    //if we don't set the camer calibration file then first we calibrate the cam
    cv::Mat camera_matrix,distortion_coefficients,image,imageUndistorted,sub;
    if(argc==1)
    {
        std::vector<cv::Mat> rvecs;
        std::vector<cv::Mat> tvecs;
        camera_calibration_example(camera_matrix,distortion_coefficients,rvecs, tvecs);
    }
    else
    { //reading camera calibration yaml file
        cv::FileStorage fs(argv[1],cv::FileStorage::READ);
        fs["camera_matrix"] >> camera_matrix;
        fs["distortion_coefficients"] >> distortion_coefficients;
        fs.release();

    }
    //we set it VideoCapture(1)
    cv::VideoCapture capture = cv::VideoCapture(0);
    while(1)
    {
        capture >> image;
        undistort(image, imageUndistorted, camera_matrix, distortion_coefficients);
        if((char)cv::waitKey(30)==(char)27)
            break;
        sub=image-imageUndistorted;
        imshow("imageUndistorted-image", sub);
        imshow("imageUndistorted", imageUndistorted);
        imshow("image", image);
    }
    capture.release();
}

int main(int argc, char** argv)
{
/**/
    cv::Mat camera_matrix;
    cv::Mat distortion_coefficients;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;

    camera_calibration_example(camera_matrix,distortion_coefficients,rvecs,tvecs);


    //undistort_example(int argc, char** argv)

}
