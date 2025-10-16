#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include<stdio.h>
using namespace cv;
using namespace std;

int iLowH = 170;
int iHighH = 179;

int iLowS = 150;
int iHighS = 255;

int iLowV = 60;
int iHighV = 255;


void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if  ( event == EVENT_LBUTTONDOWN )
    {
        cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    }
    else if  ( event == EVENT_RBUTTONDOWN )
    {
        cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    }
    else if  ( event == EVENT_MBUTTONDOWN )
    {
        cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    }
    else if ( event == EVENT_MOUSEMOVE )
    {
        cout << "Mouse move over the window - position (" << x << ", " << y << ")" << endl;
    }
}


void mouseEvent(int evt, int x, int y, int flags, void* param)
{
    char window_name[30] = "HSV Segmentation";

    Mat* image = (Mat*) param;
    if (evt == cv::EVENT_LBUTTONDOWN)
    {
//        std::cout<<"clicked point: "<<x <<","<< y<<std::endl;
//        std::cout<<"B: "<<(int)(*image).at<Vec3b>(y, x)[0]<<std::endl;
//        std::cout<<"G: "<<(int)(*image).at<Vec3b>(y, x)[1]<<std::endl;
//        std::cout<<"R: "<<(int)(*image).at<Vec3b>(y, x)[2]<<std::endl;




        Vec3b rgb=(*image).at<Vec3b>(y,x);
        int B=rgb.val[0];
        int G=rgb.val[1];
        int R=rgb.val[2];

        Mat HSV;
        Mat RGB=(*image)(Rect(x,y,1,1));
        cv::cvtColor(RGB, HSV,cv::COLOR_BGR2HSV);

        Vec3b hsv=HSV.at<Vec3b>(0,0);
        int H=hsv.val[0];
        int S=hsv.val[1];
        int V=hsv.val[2];

        iLowH=min(iLowH,H);
        iHighH=max(iHighH,H);
        cv::setTrackbarPos("LowH", "Control", iLowH);
        cv::setTrackbarPos("HighH", "Control", iHighH);



        iLowS=min(iLowS,S);
        iHighS=max(iHighS,S);
        cv::setTrackbarPos("LowS", "Control", iLowS);
        cv::setTrackbarPos("HighS", "Control", iHighS);


        iLowV=min(iLowV,V);
        iHighV=max(iHighV,V);
        cv::setTrackbarPos("LowV", "Control", iLowV);
        cv::setTrackbarPos("HighV", "Control", iHighV);


        char name[30];
        sprintf(name,"B=%d",B);
        putText((*image),name, Point(150,40) , FONT_HERSHEY_SIMPLEX, .7, Scalar(0,255,0), 2,8,false );
        std::cout<<name<<std::endl;


        sprintf(name,"G=%d",G);
        putText((*image),name, Point(150,80) , FONT_HERSHEY_SIMPLEX, .7, Scalar(0,255,0), 2,8,false );
        std::cout<<name<<std::endl;


        sprintf(name,"R=%d",R);
        putText((*image),name, Point(150,120) , FONT_HERSHEY_SIMPLEX, .7, Scalar(0,255,0), 2,8,false );
        std::cout<<name<<std::endl;


        sprintf(name,"H=%d",H);
        putText((*image),name, Point(25,40) , FONT_HERSHEY_SIMPLEX, .7, Scalar(0,255,0), 2,8,false );
        std::cout<<name<<std::endl;


        sprintf(name,"S=%d",S);
        putText((*image),name, Point(25,80) , FONT_HERSHEY_SIMPLEX, .7, Scalar(0,255,0), 2,8,false );
        std::cout<<name<<std::endl;


        sprintf(name,"V=%d",V);
        putText((*image),name, Point(25,120) , FONT_HERSHEY_SIMPLEX, .7, Scalar(0,255,0), 2,8,false );
        std::cout<<name<<std::endl;

        sprintf(name,"X=%d",x);
        putText((*image),name, Point(25,300) , FONT_HERSHEY_SIMPLEX, .7, Scalar(0,0,255), 2,8,false );
        std::cout<<name<<std::endl;

        sprintf(name,"Y=%d",y);
        putText((*image),name, Point(25,340) , FONT_HERSHEY_SIMPLEX, .7, Scalar(0,0,255), 2,8,false );
        std::cout<<name<<std::endl;


    }


}


void on_mouse( int e, int x, int y, int d, void *ptr )
{
    Point*p = (Point*)ptr;
    p->x = x;
    p->y = y;
    cout<<*p;
}

void CallBackFuncWithKey(int event, int x, int y, int flags, void* userdata)
{
    if ( flags == (EVENT_FLAG_CTRLKEY + EVENT_FLAG_LBUTTON) )
    {
        cout << "Left mouse button is clicked while pressing CTRL key - position (" << x << ", " << y << ")" << endl;
    }
    else if ( flags == (EVENT_FLAG_RBUTTON + EVENT_FLAG_SHIFTKEY) )
    {
        cout << "Right mouse button is clicked while pressing SHIFT key - position (" << x << ", " << y << ")" << endl;
    }
    else if ( event == EVENT_MOUSEMOVE && flags == EVENT_FLAG_ALTKEY)
    {
        cout << "Mouse is moved over the window while pressing ALT key - position (" << x << ", " << y << ")" << endl;
    }
}
int main( int argc, char** argv )
{
    VideoCapture cap(0); //capture the video from webcam

    if ( !cap.isOpened() )  // if not success, exit program
    {
        cout << "Cannot open the web cam" << endl;
        return -1;
    }

    namedWindow("Control", cv::WINDOW_AUTOSIZE); //create a window called "Control"



    //Create trackbars in "Control" window
    createTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
    createTrackbar("HighH", "Control", &iHighH, 179);

    createTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
    createTrackbar("HighS", "Control", &iHighS, 255);

    createTrackbar("LowV", "Control", &iLowV, 255);//Value (0 - 255)
    createTrackbar("HighV", "Control", &iHighV, 255);

    int iLastX = -1;
    int iLastY = -1;

    //Capture a temporary image from the camera
    Mat imgTmp;
    cap.read(imgTmp);

    //Create a black image with the size as the camera output
    Mat imgLines = Mat::zeros( imgTmp.size(), CV_8UC3 );;


//    while (true)
//    waitKey(30) == 27
    for(;;)
    {
        Mat imgOriginal;

        bool bSuccess = cap.read(imgOriginal); // read a new frame from video



        if (!bSuccess) //if not success, break loop
        {
            cout << "Cannot read a frame from video stream" << endl;
            break;
        }

        Mat imgHSV;

        cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

        Mat imgThresholded;

        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

        //morphological opening (removes small objects from the foreground)
        erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
        dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

        //morphological closing (removes small holes from the foreground)
        dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
        erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

        //Calculate the moments of the thresholded image
        Moments oMoments = moments(imgThresholded);

        double dM01 = oMoments.m01;
        double dM10 = oMoments.m10;
        double dArea = oMoments.m00;

        // if the area <= 10000, I consider that the there are no object in the image and it's because of the noise, the area is not zero
        if (dArea > 10000)
        {
            //calculate the position of the ball
            int posX = dM10 / dArea;
            int posY = dM01 / dArea;

            if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0)
            {
            //Draw a red line from the previous point to the current point
                line(imgLines, Point(posX, posY), Point(iLastX, iLastY), Scalar(0,0,255), 2);
            }

            iLastX = posX;
            iLastY = posY;
        }

        //  setMouseCallback("Original", CallBackFunc, NULL);
        setMouseCallback("Original", mouseEvent, &imgOriginal);

        imshow("Thresholded Image", imgThresholded); //show the thresholded image

        cv::Mat result;
        imgOriginal.copyTo(result,imgThresholded);

        imgOriginal = imgOriginal + imgLines;
        imshow("Original", imgOriginal); //show the original image

        imshow("result", result); //show the original image

        if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            cout << "esc key is pressed by user" << endl;
            break;
        }
    }

return 0;
}
