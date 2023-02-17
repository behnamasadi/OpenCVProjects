#include <opencv2/opencv.hpp>
#include "transformation.hpp"

void projectPointcloudInStereoImagePlane()
{
    cv::Mat leftCameraRotation,rightCameraRotation;
	double rollLeft, pitchLeft,  yawLeft,rollRight, pitchRight,  yawRight ,txLeft,tyLeft,tzLeft,txRight,tyRight,tzRight;

	cv::Vec3d thetaLeft,thetaRight;

	rollLeft=0 ;
	pitchLeft=+M_PI/10;
	yawLeft=0;
	
	thetaLeft[0]=rollLeft;
	thetaLeft[1]=pitchLeft;
	thetaLeft[2]=yawLeft;
	
	rollRight=0;
	pitchRight= -M_PI/10;
	yawRight= 0;
		
	thetaRight[0]=rollRight;
	thetaRight[1]=pitchRight;
	thetaRight[2]=yawRight;
	
	txLeft=-1;
	tyLeft=0.0;
    tzLeft=+4.0;
		
	txRight=1.0;
	tyRight=0.0;
    tzRight=+4.0;
	
	leftCameraRotation =eulerAnglesToRotationMatrix(thetaLeft);
	rightCameraRotation =eulerAnglesToRotationMatrix(thetaRight);
		
	cv::Mat leftCameraTranslation = (cv::Mat_<double>(3,1) <<txLeft,tyLeft,tzLeft);
	cv::Mat rightCameraTranslation = (cv::Mat_<double>(3,1) <<txRight,tyRight,tzRight);
	
	std::vector<cv::Point3d> objectPointsInWorldCoordinate;
	double X,Y,Z,radius;
	

////////////////////////////////////////creating ellipsoid in the world coordinate///////////////////////	
	double phiStepSize,thetaStepSize;
	phiStepSize=0.7;
	thetaStepSize=0.6;
	double a,b,c;
	a=2;
	b=3;
	c=1.6;
	for(double phi=-M_PI;phi<M_PI;phi=phi+phiStepSize)
	{
		for(double theta=-M_PI/2;theta<M_PI/2;theta=theta+thetaStepSize)
		{
			X=a*cos(theta)*cos(phi);
			Y=b*cos(theta)*sin(phi);
			Z=c*sin(theta);
			objectPointsInWorldCoordinate.push_back(cv::Point3d(X, Y, Z));
		}
	}

	int numberOfPixelInHeight,numberOfPixelInWidth;
	double heightOfSensor, widthOfSensor;
	double focalLength=2.0;
	double mx, my, U0, V0;
	numberOfPixelInHeight=600;
	numberOfPixelInWidth=600;
	
	heightOfSensor=10;
	widthOfSensor=10;
	
	my=(numberOfPixelInHeight)/heightOfSensor ;
	U0=(numberOfPixelInHeight)/2 ;

	mx=(numberOfPixelInWidth)/widthOfSensor; 
	V0=(numberOfPixelInWidth)/2;

	cv::Mat K = (cv::Mat_<double>(3,3) <<
	focalLength*mx, 0, V0,
	0,focalLength*my,U0,
	0,0,1);
	
	std::vector<cv::Point2d> imagePointsLeftCamera,imagePointsRightCamera;
    cv::projectPoints(objectPointsInWorldCoordinate, leftCameraRotation.inv(), -leftCameraTranslation, K, cv::noArray(), imagePointsLeftCamera);
    cv::projectPoints(objectPointsInWorldCoordinate, rightCameraRotation.inv(), -rightCameraTranslation, K, cv::noArray(), imagePointsRightCamera);

////////////////////////////////////////////////storing images from right and left camera//////////////////////////////////////////////

	std::string fileName;
	cv::Mat rightImage,leftImage;
	int U,V;
	leftImage=cv::Mat::zeros(numberOfPixelInHeight,numberOfPixelInWidth,CV_8UC1);

	for(std::size_t i=0;i<imagePointsLeftCamera.size();i++)
	{
		V=int(imagePointsLeftCamera.at(i).x);
		U=int(imagePointsLeftCamera.at(i).y);
		leftImage.at<char>(U,V)=(char)255;
	}

	fileName=std::string("imagePointsLeftCamera")+std::to_string(focalLength)+ std::string("_.jpg");
	cv::imwrite(fileName,leftImage);

	rightImage=cv::Mat::zeros(numberOfPixelInHeight,numberOfPixelInWidth,CV_8UC1);
	for(std::size_t i=0;i<imagePointsRightCamera.size();i++)
	{
		V=int(imagePointsRightCamera.at(i).x);
		U=int(imagePointsRightCamera.at(i).y);
		rightImage.at<char>(U,V)=(char)255;
	}

	fileName=std::string("imagePointsRightCamera")+std::to_string(focalLength)+ std::string("_.jpg");
	cv::imwrite(fileName,rightImage);

}

int main()
{
    projectPointcloudInStereoImagePlane();
}
