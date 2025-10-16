#include <opencv2/opencv.hpp>


/*
OpenCV camera coordinate:
 
                  Z
                ▲
               /
              /
             /1 2 3 4     x or u means column
            |------------ ⯈
           1|           
           2|       
           3|           
           4|           
            | y or v means row
            ⯆




In OpenCV, Point(x=column,y=row). For instance the point in the following image can be accessed with

    X                      
    --------column---------►
    | Point(0,0) Point(1,0) Point(2,0) Point(3,0)
    | Point(0,1) Point(1,1) Point(2,1) Point(3,1)
    | Point(0,2) Point(1,2) Point(2,2) Point(3,2)
  y |
   row
    |
    |
    ▼

    However if you access an image directly, the order is mat.at<type>(row,column). So the following will return the same value:
    mat.at<type>(row,column) 
    mat.at<type>(cv::Point(column,row))

    X                      
    --------column---------►
    | mat.at<type>(0,0) mat.at<type>(0,1) mat.at<type>(0,2) mat.at<type>(0,3)
    | mat.at<type>(1,0) mat.at<type>(1,1) mat.at<type>(1,2) mat.at<type>(1,3)
    | mat.at<type>(2,0) mat.at<type>(2,1) mat.at<type>(2,2) mat.at<type>(2,3)
  y |
   row
    |
    |
    ▼
*/    
void imageCoordinateVSPoint(int argc, char** argv)
{
    cv::Mat img=cv::imread(argv[1],cv::IMREAD_GRAYSCALE );
    //img1 type will be CV_8UC1
    int row, column;

    row=50;
    column=200;

    std::cout<<static_cast<unsigned>(img.at<uchar>(row,column))    <<std::endl;
    std::cout<<static_cast<unsigned>(img.at<uchar>( cv::Point(column,row))     )<<std::endl;
}

void coordinateSystemDisplay()
{

    int blue, green, red;
    blue=255;
    green=255;
    red=255;
    int numberOfRows, numberOfCols;

    numberOfRows=480;
    numberOfCols=640;

    cv::Mat img = cv::Mat::zeros(numberOfRows,numberOfCols, CV_8UC3);
    std::cout<<"number of rows:" <<img.rows  <<std::endl;
    std::cout<<"number of cols:" <<img.cols  <<std::endl;

    int row, column;

    row=140;
    column=10;


    for(int column=0;column<img.cols;column++)
    {
        //img.at<cv::Vec3b>( row, column)=cv::Vec3b(blue,green,red);
        img.at<cv::Vec3b>( cv::Point( column,row))=cv::Vec3b(blue,green,red);
        cv::imwrite( "row_"+std::to_string(row) + "_cols_" +std::to_string(column) +".png"  ,img);
    }


    // for(int row=0;row<img.rows;row++)
    // {
    //     img.at<cv::Vec3b>( cv::Point( column,row))=cv::Vec3f(blue,green,red);
    //     //img.at<cv::Vec3b>( row, column)=cv::Vec3f(blue,green,red);
    //     cv::imwrite( "row_"+std::to_string(row) + "_cols_" +std::to_string(column) +".png"  ,img);
    // }


}

int main(int argc, char** argv)
{
    coordinateSystemDisplay();
}
