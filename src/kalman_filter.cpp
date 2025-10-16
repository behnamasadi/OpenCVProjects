//http://www.morethantechnical.com/2011/06/17/simple-kalman-filter-for-tracking-using-opencv-2-2-w-code/
//https://ros-developer.com/2019/04/10/kalman-filter-explained-with-python-code-from-scratch/
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <fstream> 

#include <Eigen/Dense>
 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>



struct normal_random_variable
{
    normal_random_variable(Eigen::MatrixXd const& covar)  : normal_random_variable(Eigen::VectorXd::Zero(covar.rows()), covar)
    {}

    normal_random_variable(Eigen::VectorXd const& mean, Eigen::MatrixXd const& covar) : mean(mean)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
        transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    Eigen::VectorXd mean;
    Eigen::MatrixXd transform;

    Eigen::VectorXd operator()() const
    {
        static std::mt19937 gen{ std::random_device{}() };
        static std::normal_distribution<> dist;

        return mean + transform * Eigen::VectorXd{ mean.size() }.unaryExpr([&](auto x) { return dist(gen); });
    }
};

struct mouse_info_struct { int x,y; };

struct mouse_info_struct mouse_info = {-1,-1}, last_mouse;
 
 
void on_mouse(int event, int x, int y, int flags, void* param) {
	{
		last_mouse = mouse_info;
		mouse_info.x = x;
		mouse_info.y = y;
	}
}
 
// plot points
#define drawCross( center, color, d )                                 \
cv::line( img, cv::Point( center.x - d, center.y - d ),                \
cv::Point( center.x + d, center.y + d ), color, 2, cv::LINE_AA, 0); \
cv::line( img, cv::Point( center.x + d, center.y - d ),                \
cv::Point( center.x - d, center.y + d ), color, 2, cv::LINE_AA, 0 )
 

void save2DPointsInCSVData(std::string fileName, std::vector<cv::Point> points)
{
    std::ofstream file(fileName,std::ofstream::ate );
    file<<"x,y"<<std::endl;
    for (std::size_t i = 0; i < points.size(); i++)
    {
        file<<points[i].x << ","<<points[i].y <<std::endl;
    }
    file.close();
} 
 
int main (int argc, char * const argv[]) 
{
    int k=5;

    int size = 2;
    Eigen::VectorXd  mean(size);
    mean<<0,0;
    Eigen::MatrixXd covar(size,size);
    covar << k*1, 0,
            0, k*1;

    normal_random_variable sample {mean, covar };


    std::vector<cv::Point> groundTruth,kalmanv,measurmens;



    cv::Mat img(500, 500, CV_8UC3);
    cv::KalmanFilter KF(4, 2, 0);
    cv::Mat_<float> state(4, 1); /* (x, y, Vx, Vy) */
    cv::Mat processNoise(4, 1, CV_32F);
    cv::Mat_<float> measurement(2,1); measurement.setTo(cv::Scalar(0));
    char code = char(-1);
	
    cv::namedWindow("Mouse Tracking with Kalman Filter");
    cv::setMouseCallback("Mouse Tracking with Kalman Filter", on_mouse, nullptr);
    double delta_t=1/20.0;
	
    for(;;)
    {
		if (mouse_info.x < 0 || mouse_info.y < 0) {
            imshow("Mouse Tracking with Kalman Filter", img);
            cv::waitKey(30);
			continue;
		}
        cv::Mat transitionMatrix=(cv::Mat_<float>(4, 4) << 1,0,delta_t,0,   0,1,0,delta_t,  0,0,1,0,  0,0,0,1);
        KF.transitionMatrix = transitionMatrix;
		
        setIdentity(KF.measurementMatrix);
        cv::setIdentity(KF.processNoiseCov, cv::Scalar::all(1e-0));
        cv::setIdentity(KF.measurementNoiseCov, cv::Scalar::all(k*1e-0));
        cv::setIdentity(KF.errorCovPost, cv::Scalar::all(.2));
        cv::setIdentity(KF.errorCovPre,cv::Scalar::all(.1));
 
        measurmens.clear();
        groundTruth.clear();
		kalmanv.clear();
        std::cout<< "measurementMatrix"<<std::endl;
        std::cout<<KF.measurementMatrix<<std::endl;
		
        for(;;)
        {
            std::cout<< "processNoiseCov"<<std::endl;
            std::cout<< KF.processNoiseCov<<std::endl;
 
            std::cout<< "measurementNoiseCov"<<std::endl;
            std::cout<< KF.measurementNoiseCov<<std::endl;
 
            std::cout<<"State Prior (before calling predict function):" <<std::endl;
            std::cout<<KF.statePre <<std::endl;
 
            std::cout<<"Cov Prior (before calling predict function):" <<std::endl;
            std::cout<<KF.errorCovPre <<std::endl;
 
            std::cout<<"Cov Posterior (before calling predict function):" <<std::endl;
            std::cout<<KF.errorCovPost <<std::endl;
 
            //KF.controlMatrix
            std::cout<<"My State Prediction:" <<std::endl;
            std::cout<<KF.transitionMatrix*KF.statePost<<std::endl;
            std::cout<<"My Cov Prediction:" <<std::endl;
            std::cout<<KF.transitionMatrix*KF.errorCovPost*KF.transitionMatrix.t()+KF.processNoiseCov<<std::endl;
 
            cv::Mat prediction = KF.predict();
            cv::Point predictPt(prediction.at<float>(0),prediction.at<float>(1));
 
            std::cout<<"OpenCV Prediction:" <<std::endl;
 
            std::cout<<"State Prior:" <<std::endl;
            std::cout<<KF.statePre <<std::endl;
 
            std::cout<<"OpenCV Cov Prior:" <<std::endl;
            std::cout<<KF.errorCovPre <<std::endl;
 
 
            measurement(0) = mouse_info.x+sample()(0);
            measurement(1) = mouse_info.y+sample()(1);
 
            cv::Point groundtruth(mouse_info.x,mouse_info.y);
            groundTruth.push_back(groundtruth);
 
 
            std::cout<<"Ground Truth:" <<std::endl;
            std::cout<< mouse_info.x<<" , "<<mouse_info.y <<std::endl;
 
            cv::Point measPt(measurement(0),measurement(1));
            measurmens.push_back(measPt);
 
 
            cv::Mat estimated = KF.correct(measurement);
            cv::Point statePt(estimated.at<float>(0),estimated.at<float>(1));
			kalmanv.push_back(statePt);
 
            std::cout<<"My State Posterior:" <<std::endl;
            std::cout<<KF.statePre+KF.gain*(measurement-KF.measurementMatrix*KF.statePre) <<std::endl;
 
            std::cout<<"My Cov Posterior:" <<std::endl;
            std::cout<<(cv::Mat::eye(4,4, CV_32F) - KF.gain*KF.measurementMatrix)*KF.errorCovPre <<std::endl;
 
            std::cout<<"Opencv State Posterior:" <<std::endl;
            std::cout<<KF.statePost <<std::endl;
 
            std::cout<<"Opencv Cov Posterior:" <<std::endl;
            std::cout<<KF.errorCovPost <<std::endl;
 
            std::cout<<"-----------------------------------------------" <<std::endl;
 
            img = cv::Scalar::all(0);
            drawCross( statePt, cv::Scalar(255,255,255), 5 );
            drawCross( measPt, cv::Scalar(0,0,255), 5 );
            for (std::size_t i = 0; i < groundTruth.size()-1; i++)
            {
                line(img, groundTruth[i], groundTruth[i+1], cv::Scalar(0,255,0), 1);
			}
            for (std::size_t i = 0; i < kalmanv.size()-1; i++)
            {
                line(img, kalmanv[i], kalmanv[i+1], cv::Scalar(255,0,0), 1);
			}
            for (std::size_t i = 0; i < measurmens.size()-1; i++)
            {
                line(img, measurmens[i], measurmens[i+1], cv::Scalar(0,255,255), 1);
            }
            cv::putText(img,"Noisy Measurements",cv::Point(10,10),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(0,255,255) 	);
            cv::putText(img,"Real Mouse Position(ground truth)",cv::Point(10,25),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(0,255,0));
            cv::putText(img,"Kalman Sate",cv::Point(10,35),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,0,0));
 
 
            imshow( "Mouse Tracking with Kalman Filter", img );
            code = char(cv::waitKey(1000.0*delta_t));
			
            if( code > 0 )
                break;
        }
        if( code == 27 || code == 'q' || code == 'Q' )
            break;
    }
 
    std::cout<<"saving points into CSV files" <<std::endl;

    save2DPointsInCSVData("groundTruth.csv", groundTruth);
    save2DPointsInCSVData("kalmanv.csv", kalmanv);
    save2DPointsInCSVData("measurmens.csv", measurmens);


    

 
    return 0;
}