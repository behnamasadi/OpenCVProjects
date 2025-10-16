#include <opencv2/opencv.hpp>
//#include "opencv2/imgproc.hpp"
/*
remapping is taking pixels from one place in the image and locating them in another position in a new image.
since there will not always be a one-to-one-pixel correspondence between source and destination images some interpolation might be necessary,.


remapped_image(x,y)=source_image(mapping_function(x,y))

*/

void remapExample(int argc, char **argv)
{
    if( argc != 2)
    {
        std::cout <<" Usage: ./main <image> \n"<< std::endl;
        return ;
    }

    cv::Mat source_image, remapped_image, warp_rotate_dst;
    source_image = cv::imread( argv[1], cv::IMREAD_ANYCOLOR);

  
    
    //cv::Mat map_x(source_image.size(), source_image.type());
    /*
    that map_y and map_x are both of the same size as source_imagec
    map_x: The mapping function in the x direction. It is equivalent to the first component of remapped_image(x,y)
    map_y: The mapping function in the y direction. It is equivalent to the second component of remapped_image(x,y)
    */


    cv::Mat map_x(source_image.size(), CV_32FC1);
    cv::Mat map_y(source_image.size(), CV_32FC1);
    
    for(auto i=0;i<source_image.rows;i++)
    {
        for(auto j=0;j<source_image.rows;j++)
        {
            map_x.at<float>(i, j) = (float)(map_x.cols - j);    
            map_y.at<float>(i, j) = (float)(map_x.rows - i);
        }
    }

    std::cout<< "map_x\n"<<  map_x <<std::endl;
    std::cout<< "map_y\n"<<map_y <<std::endl;

    cv::remap( source_image, remapped_image,map_x,map_y,cv::INTER_LINEAR, cv::BORDER_CONSTANT,cv::Scalar(0, 0, 0) );

    std::string source_window = "source_image";
    std::string remapped_window = "remapped_image";

    cv::namedWindow(source_window, cv::WINDOW_AUTOSIZE);
    cv::namedWindow(remapped_window, cv::WINDOW_AUTOSIZE);

    cv::imshow( source_window, source_image );
    cv::imshow( remapped_window, remapped_image );
    
    cv::waitKey(0);


}


int main(int argc, char **argv)
{
    remapExample(argc, argv);
}