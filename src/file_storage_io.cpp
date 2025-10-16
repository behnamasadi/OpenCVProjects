#include <opencv2/opencv.hpp>

void readingCameraMatrix()
{

    std::string camera_calibration_path="../data/front_webcam.yml";
    cv::FileStorage fs(camera_calibration_path,cv::FileStorage::READ);
    cv::Mat camera_matrix, distortion_coefficient;
    fs["camera_matrix"]>>camera_matrix;
    fs["distortion_coefficients"]>>distortion_coefficient;

    std::cout<<"Camera Matrix:" <<std::endl;

    std::cout<<"Fx: " <<camera_matrix.at<double>(0,0) <<std::endl;
    std::cout<<"Fy: " <<camera_matrix.at<double>(1,1) <<std::endl;

    std::cout<<"Cx: " <<camera_matrix.at<double>(0,2) <<std::endl;
    std::cout<<"Cy: " <<camera_matrix.at<double>(1,2) <<std::endl;
    std::cout<< "Distortion Coefficient:"<<std::endl;

    std::cout<<"K1: "<<distortion_coefficient.at<double>(0,0) <<std::endl;
    std::cout<<"K2: "<<distortion_coefficient.at<double>(0,1) <<std::endl;
    std::cout<<"P1: "<<distortion_coefficient.at<double>(0,2) <<std::endl;
    std::cout<<"P2: "<<distortion_coefficient.at<double>(0,3) <<std::endl;
    std::cout<<"K3: "<<distortion_coefficient.at<double>(0,4) <<std::endl;

}



int main(){}
