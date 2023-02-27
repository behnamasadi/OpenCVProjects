#include <opencv2/opencv.hpp>

/*
1) Depth

uchar       CV_8U   0
char        CV_8S   1
ushort      CV_16U  2
short       CV_16S  3
int	        CV_32S  4
float       CV_32F  5
double	    CV_64F  6
            CV_USRTYPE1 7


2) Type
    CV_<bit_depth>(S|U|F)C<number_of_channels>
    elements type (uchar,short,int,float,double)
    CV_8UC1 means an 8-bit unsigned single channel
    CV_32FC3 means a 32-bit float matrix with three

    CV_32F is float!
    CV_64F is double!

              C1   C2    C3    C4
    CV_8U     0    8     16    24
    CV_8S     1    9     17    25
    CV_16U    2    10    18    26
    CV_16S    3    11    19    27
    CV_32S    4    12    20    28
    CV_32F    5    13    21    29
    CV_64F    6    14    22    30


*/

void createMatrix() {

  int rows, cols;
  rows = 600;
  cols = 800;

  // 1)
  cv::Mat img1 = cv::Mat::zeros(rows, cols, CV_64FC3) + 0.5;
  // or
  cv::Mat img10 = cv::Mat::zeros(rows, cols, CV_64FC(3)) + 0.5;

  // setting channel values: B: 0.5, G: 0,  R: 1
  cv::Mat img11 = cv::Mat(rows, cols, CV_64FC(3), cv::Scalar(0.5, 0, 1));

  // 2)
  cv::Mat dst = cv::Mat::zeros(img1.size(), img1.type());

  // 3) create matrix in several step;
  cv::Mat img3;
  img3.create(rows, cols, CV_32FC1);

  // 4) cv::DataType<double>::type
  cv::Mat cameraMatrix1(3, 3, cv::DataType<double>::type);
  std::cout << cameraMatrix1 << std::endl;

  // 5) cv::Mat_<double>(3,3)
  cv::Mat cameraMatrix2 = (cv::Mat_<double>(3, 3) << 1, 2, 3, 4, 5, 6);
  std::cout << cameraMatrix2 << std::endl;

  // 6) Matrices with More than Two Dimensions
  // Mat is usually used for 2D matrices. we want multi-dimension matrices:

  std::vector<int> dims = {5, 3, 7};
  cv::Mat mat(dims, CV_32FC1);
  std::cout << mat.at<float>(0, 0, 0);

  // 7 Matrices from existing data
  float data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  cv::Mat mat_data = cv::Mat(2, 5, CV_32F, data);

  cv::namedWindow("window", cv::WINDOW_AUTOSIZE);
  cv::imshow("window", img1);
  cv::waitKey(0);
}

void matrixVectorConversion() {
  std::vector<double> values = {1};

  cv::Mat MatrixofRawData;
  int NumberofRows = 2;
  int NumberofCols = 3;
  MatrixofRawData.create(NumberofRows, NumberofCols, CV_32FC1);

  std::vector<float> RowVector;
  RowVector.push_back(0.2);
  RowVector.push_back(1.2);
  RowVector.push_back(4.2);
  MatrixofRawData.push_back(RowVector);
  // MatrixofRawData.convertTo();
  cv::Mat MatFromVector(RowVector, true);
  std::cout << "MatFromVector: " << MatFromVector << std::endl;

  // createing image, merging, spliting channels
}

void matrixFromVectorOfCVPoints() {
  std::vector<cv::Point> pixelValue;
  pixelValue.push_back(cv::Point(320, 240));
  pixelValue.push_back(cv::Point(326.4, 249.6));
  pixelValue.push_back(cv::Point(332.8, 249.6));
  pixelValue.push_back(cv::Point(339.2, 249.6));
  pixelValue.push_back(cv::Point(332.8, 254.4));
  pixelValue.push_back(cv::Point(332.8, 259.2));

  cv::Mat pointInCamera = cv::Mat(pixelValue);

  std::cout << pointInCamera << std::endl;
  std::cout << "rows: " << pointInCamera.rows << std::endl;
  std::cout << "cols: " << pointInCamera.cols << std::endl;
  std::cout << "channels: " << pointInCamera.channels() << std::endl;

  std::vector<cv::Point3d> vertices;
  vertices.push_back(cv::Point3d(0, 0, 1));
  vertices.push_back(cv::Point3d(2, 1, 1));
  vertices.push_back(cv::Point3d(1, 2, 1));
  vertices.push_back(cv::Point3d(2, 2, 1));
  vertices.push_back(cv::Point3d(3, 2, 1));
  vertices.push_back(cv::Point3d(2, 3, 1));
  vertices.push_back(cv::Point3d(2, 4, 1));

  // reshape(1)  make Nx3 1-channel matrix out of Nx1 3-channel.
  // t() transpose the Nx3 matrix.

  cv::Mat pointInWorld = cv::Mat(vertices).reshape(1).t();
  std::cout << pointInWorld << std::endl;
  std::cout << "rows: " << pointInWorld.rows << std::endl;
  std::cout << "cols: " << pointInWorld.cols << std::endl;
  std::cout << "channels: " << pointInWorld.channels() << std::endl;
}

void matrixOperations() {
  cv::Mat m1 = cv::Mat::zeros(3, 3, CV_64FC1);
  cv::Mat m2 = cv::Mat::ones(3, 3, CV_64FC1);

  std::cout << "m1" << std::endl;
  std::cout << m1 << std::endl;

  std::cout << "m2" << std::endl;
  std::cout << m2 << std::endl;

  std::cout << "(m2*2)" << std::endl;
  std::cout << (m2 * 2) << std::endl;

  std::cout << "m2*m1" << std::endl;
  std::cout << m2 * m1 << std::endl;

  std::cout << "m1+m2.t()" << std::endl;
  std::cout << m1 + m2.t() << std::endl;

  std::cout << "m1-m2.t()" << std::endl;
  std::cout << m1 - m2.t() << std::endl;
}

void accessingMatrixElements() {
  int rows, cols, i, j;
  rows = 600;
  cols = 800;
  i = rows / 2;
  j = cols / 2;

  cv::Point point = cv::Point(i, j);
  cv::Mat img1, img2, img3;

  // CV_8UC4 means unsinged char (8 bit 0-255) and 4 channel, so to access every
  // pixel we use cv::Vec4b
  img1.create(rows, cols, CV_8UC4);

  // CV_64FC1 means double  and 4 channel, so to access every pixel we use
  // cv::Vec4d
  img2.create(rows, cols, CV_64FC4);
  // cv::Vec4d;
  cv::Vec4d pixel_values_bgra = img2.at<cv::Vec4d>(point);
  double pixel_value_b = img2.at<cv::Vec4d>(point)[0];

  // CV_64FC1 means floar  and 1 channel, so to access every pixel we use
  // cv::Vec4d
  img3.create(rows, cols, CV_32FC1);
  // float pixel=img3.at<float>(point);
  img3.at<float>(i, j) = 0.5;
  float pixel = img3.at<float>(i, j);

  cv::namedWindow("window", cv::WINDOW_AUTOSIZE);
  cv::imshow("window", img3);
  cv::waitKey(0);
}

void readWriteImage() {
  std::string imageDir = "../images/";
  std::string img1FileName = "lena.jpg";
  std::string windowName = "window";
  cv::Mat img1;
  img1 = cv::imread(imageDir + img1FileName, cv::IMREAD_COLOR);

  // img1 type will be CV_8UC3
  std::cout << img1.type() << std::endl;

  // img1 type will be CV_8UC1
  img1 = cv::imread(imageDir + img1FileName, cv::IMREAD_GRAYSCALE);
  std::cout << img1.type() << std::endl;

  /*
  cv::minMaxIdx finds the minimum and maximum element values and their
  positions, does not work with multi-channel arrays, use Mat::reshape first to
  reinterpret the array as single-channel. Or you may extract the particular
  channel using either extractImageCOI , or mixChannels , or split .
  */
  std::vector<int> minIx(3), maxIx(3);

  double minValue, maxValue;
  cv::minMaxIdx(img1, &minValue, &maxValue, &minIx[0], &maxIx[0]);

  std::cout << "min value: " << minValue << " max value: " << maxValue
            << "min ix: [" << minIx[0] << " " << minIx[1] << " " << minIx[2]
            << "] max id [" << maxIx[0] << " " << maxIx[1] << " " << maxIx[2]
            << "]" << std::endl;

  // if you need to read it in double, you have to convert it:
  img1.convertTo(img1, CV_64FC1, 1 / 255.0);
  std::cout << img1.type() << std::endl;
  cv::minMaxIdx(img1, &minValue, &maxValue, &minIx[0], &maxIx[0]);
  std::cout << "min value: " << minValue << " max value: " << maxValue
            << "min ix: [" << minIx[0] << " " << minIx[1] << " " << minIx[2]
            << "] max id [" << maxIx[0] << " " << maxIx[1] << " " << maxIx[2]
            << "]" << std::endl;

  cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
  cv::imshow(windowName, img1);
  cv::waitKey(0);
}

void matrixConversion() {
  // convertTo
  // cv::cvtColor
}

void drawingFunsctionAndCoordinate() {
  /*
  image's points in opencv has the follow index:
      (0,0) (1,0) (2,0) 3,0)
      (0,1) (1,1) (2,1) 3,1)
      (0,2) (1,2) (2,2) 3,2)

          X           (cols,0)
          -------------►
          |
        y |
          |
  (0,rows)▼           (cols,rows)

*/
  int blue, green, red;
  blue = 255;
  green = 255;
  red = 255;

  cv::Mat img = cv::Mat::zeros(400, 600, CV_32FC3);
  cv::namedWindow("WorkingwitDrawingcommands", cv::WINDOW_AUTOSIZE);

  // draw a box with red lines of width 1 between (0,100) and (200,200)
  cv::rectangle(img, cv::Point(0, 100), cv::Point(200, 200),
                cv::Scalar(blue, 0, red), 1);

  // draw a circle at (300,300) with a radius of 20. Use green lines of width 1
  cv::circle(img, cv::Point(300, 100), 20, cv::Scalar(0, green, 0), 1);

  // Draw a line segment:

  // draw a green line of width 1 between (100,100) and (200,200)
  cv::line(img, cv::Point(100, 100), cv::Point(200, 200), cv::Scalar(0, 255, 0),
           1);

  // Draw a set of polylines:
  cv::Point curve1[] = {cv::Point(10, 10), cv::Point(10, 100),
                        cv::Point(100, 100), cv::Point(100, 10)};
  cv::Point curve2[] = {cv::Point(30, 30), cv::Point(30, 130),
                        cv::Point(130, 130), cv::Point(130, 30),
                        cv::Point(150, 10)};
  const cv::Point *curveArr[2] = {curve1, curve2};
  int nCurvePts[2] = {4, 5};
  int nCurves = 2;
  int isCurveClosed = 1;
  int lineWidth = 1;
  cv::polylines(img, curveArr, nCurvePts, nCurves, isCurveClosed,
                cv::Scalar(0, 255, 255), lineWidth);

  // Draw a set of filled polygons:

  cv::fillPoly(img, curveArr, nCurvePts, nCurves, cv::Scalar(0, 255, 255));

  // Add text:

  double fontScale = 1.0;
  cv::putText(img, "My comment", cv::Point(200, 400), cv::FONT_HERSHEY_SIMPLEX,
              fontScale, cv::Scalar(255, 255, 0));
  //	Other possible fonts:
  //
  //	CV_FONT_HERSHEY_SIMPLEX, CV_FONT_HERSHEY_PLAIN,
  //	CV_FONT_HERSHEY_DUPLEX, CV_FONT_HERSHEY_COMPLEX,
  //	CV_FONT_HERSHEY_TRIPLEX, CV_FONT_HERSHEY_COMPLEX_SMALL,
  //	CV_FONT_HERSHEY_SCRIPT_SIMPLEX, CV_FONT_HERSHEY_SCRIPT_COMPLEX,

  // polylines(img, vert, true, Scalar(255)); // or perhaps 0

  cv::imshow("WorkingwitDrawingcommands", img);
  cv::waitKey(0);
}

void manipulatingImageChannels() {

  // 1) first way
  cv::Mat img(600, 800, CV_64FC3); // declare three channels image
  // "channels" is a vector of 3 Mat arrays:
  std::vector<cv::Mat> channels(3);

  // split img: split will always copy the data, since it's creating new
  // matrices
  cv::split(img, channels);
  // get the channels (follow BGR order in OpenCV), modify channel, then merge
  channels[0] = channels[0] + 0.5;

  // merge doesn't allocate any new memory,
  cv::merge(channels, img);

  // 2) second way
  img.setTo(cv::Scalar(0.5, 0, 1));

  // 3) third way

  if (img.isContinuous()) {
  }

  cv::namedWindow("window", cv::WINDOW_AUTOSIZE);
  cv::imshow("window", img);
  cv::waitKey(0);
}

void displayingVideo() {
  cv::VideoCapture cap;
  cap.open("../videos/cup.mp4");

  if (!cap.isOpened()) // check if we succeeded
    return;

  cv::namedWindow("xing", 1);
  std::cout << "number of frames:" << cap.get(cv::CAP_PROP_FRAME_COUNT)
            << " frame rate: " << cap.get(cv::CAP_PROP_FPS)
            << " resolution is: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << ","
            << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;

  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640 * 2);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480 * 2);

  for (;;) {
    cv::Mat frame;
    cap >> frame; // get a new frame from camera
    if (frame.empty())
      break;
    imshow("xing", frame);
    if (cv::waitKey(20) >= 0)
      break;
    std::cout << "frame is at time: " << cap.get(cv::CAP_PROP_POS_MSEC)
              << std::endl;
  }
  // moving the player to arbitrart frame
  int start = cap.get(cv::CAP_PROP_FRAME_COUNT) / 2;
  cap.set(cv::CAP_PROP_POS_FRAMES, start);

  for (int i = start; i < cap.get(cv::CAP_PROP_FRAME_COUNT); i++) {
    cv::Mat frame;
    cap >> frame;

    if (frame.empty())
      break;

    imshow("xing", frame);
    if (cv::waitKey(20) >= 0)
      break;
    std::cout << "frame is at time: " << cap.get(cv::CAP_PROP_POS_MSEC)
              << std::endl;
  }
}

void writingVideo() {
  cv::VideoCapture cap;
  cap.open("../videos/xing.mp4");

  cv::Mat frame;

  cap >> frame;

  bool isColor = (frame.type() == CV_8UC3);

  cv::VideoWriter output;
  std::string vid_output = "vid.mp4";
  cv::Size videoFrameSize;

  videoFrameSize.height = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  videoFrameSize.width = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

  std::cout << "frame rate: " << cap.get(cv::CAP_PROP_FPS) << std::endl;

  std::cout << "video frame size is: " << videoFrameSize << std::endl;

  // For H.264  use AVC, which would look like this:
  //int codec = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
  //https://learn.microsoft.com/en-us/windows/win32/medfound/video-fourccs

  int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

  output.open(vid_output, codec, cap.get(cv::CAP_PROP_FPS), videoFrameSize,
              isColor);

  if (!output.isOpened()) {
    std::cerr << "Could not open the output video file for write\n";
    return;
  }

  int start = cap.get(cv::CAP_PROP_FRAME_COUNT) / 2;
  cv::namedWindow("xing", 1);
  for (int i = start; i < cap.get(cv::CAP_PROP_FRAME_COUNT); i++) {
    cap >> frame;

    cv::imshow("xing", frame);
    if (cv::waitKey(20) >= 0)
      break;
    if (frame.empty())
      break;
    // std::cout<<frame.size() <<std::endl;
    //output.write(frame);
    output<<frame;
  }
  output.release();

  // cv::CAP_PROP_FPS =5,
  // cv::CAP_PROP_POS_FRAMES =1,
  // cv::CAP_PROP_POS_MSEC =
}

void getVideoFromCam() {
  cv::VideoCapture webCam(0); // open the default camera
  webCam.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  webCam.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  if (!webCam.isOpened()) // check if we succeeded
    return;

  cv::namedWindow("camera", 1);
  for (;;) {
    cv::Mat frame;
    webCam >> frame;
    cv::imshow("camera", frame);
    if (cv::waitKey(200) >= 0)
      break;
  }
}

void matStructure() {

  // Mat is to treat it like a smart pointer (like shared_pt
  cv::Mat img1, img2;
  int rows, cols;
  rows = 600;
  cols = 800;
  // img1.create(rows, cols,CV_64FC3);
  img1.create(rows, cols, CV_64FC3);
  cv::Mat M(rows, cols, CV_64FC3, cv::Scalar(1, 1, 0));

  /*
  img2 and img1 share one memory part for their internal matrix data,
  any change you make to the matrix data of img1 or img2 will happen to another
  one
  */
  img2 = img1;

  // To create a new clone independent from m1, we can use clone() function:
  cv::Mat img3 = img1.clone();
  // if you already have img4, use copyTo():
  cv::Mat img4;
  img1.copyTo(img4);

  std::cout << "Number of Channels: " << img1.channels() << std::endl;
  std::cout << "Number of dims: " << img1.dims << std::endl;

  // According to the above table CV_64F=6
  std::cout << "Matrix data depth is: " << img1.depth() << std::endl;
  // According to the above table CV_64FC3=22
  std::cout << "Matrix data type is: " << img1.type() << std::endl;

  std::cout << "rows: " << img1.rows << std::endl;
  std::cout << "cols: " << img1.cols << std::endl;
  std::cout << "height: " << img1.size().height << std::endl;
  std::cout << "width: " << img1.size().width << std::endl;

  // total() returns the number number of pixels, if channel is more than 1,
  // each pixel is again an array
  std::cout << "total()= Number of Elements(Pixels): " << img1.total()
            << ", which is rows*cols=" << img1.rows * img1.cols << std::endl;

  // returns the matrix element size in bytes. For example, if the matrix type
  // is CV_16SC3 ,  the method returns 3*sizeof(short) or 6
  std::cout << "size of matrix element size * by number of channels : "
            << img1.elemSize() << std::endl;

  // matrix element channel size in bytes, that is, it ignores the number of
  // channels. For example, if the matrix type is CV_16SC3 , the method returns
  // sizeof(short) or 2
  std::cout << "size of matrix element size : " << img1.elemSize1()
            << std::endl;

  std::cout << "data size: " << img1.total() * img1.elemSize() << std::endl;

  std::cout << "Number of Bytes: "
            << img1.rows * img1.cols * img1.channels() * sizeof(double)
            << std::endl;

  // cv::Mat::data pointer,
  double *input = (double *)(img1.data);

  /*reshape
  Mat Mat::reshape(int cn, int rows=0) const

  cn – New number of channels. If the parameter is 0, the number of channels
  remains the same.

  rows – New number of rows. If the parameter is 0, the number of rows remains
  the same.

  mat = [a b c d]
  mat.reshape(0,2)
  [a b; c d]
  */

  // rowRange

  cv::namedWindow("window", cv::WINDOW_AUTOSIZE);
  cv::imshow("window", M);
  cv::waitKey(0);
}

void scalarValues() {
  std::cout << cv::Scalar::all(1.0) << std::endl;

  std::cout << cv::Scalar(1) << std::endl;

  std::cout << cv::Scalar(1, 1) << std::endl;

  cv::Mat m(100, 100, CV_8UC3);
  m = cv::Scalar(5, 10, 15);

  cv::Mat M(7, 7, CV_32FC2, cv::Scalar(1, 3));
}

cv::Mat createMat(unsigned char *rawData, unsigned int dimX,
                  unsigned int dimY) {
  // No need to allocate outputMat here
  cv::Mat outputMat;

  // Build headers on your raw data
  cv::Mat channelR(dimY, dimX, CV_8UC1, rawData);
  cv::Mat channelG(dimY, dimX, CV_8UC1, rawData + dimX * dimY);
  cv::Mat channelB(dimY, dimX, CV_8UC1, rawData + 2 * dimX * dimY);

  // Invert channels,
  // don't copy data, just the matrix headers
  std::vector<cv::Mat> channels{channelB, channelG, channelR};

  // Create the output matrix
  cv::merge(channels, outputMat);

  return outputMat;
}

void draw_xyz_frame_over_image(cv::Mat image, cv::Point2f reference_point,
                               std::vector<cv::Point2f> end_points,
                               int thickness = 5) {
  cv::line(image, reference_point, end_points.at(0), cv::Scalar(255, 0, 0),
           thickness);
  cv::line(image, reference_point, end_points.at(1), cv::Scalar(0, 255, 0),
           thickness);
  cv::line(image, reference_point, end_points.at(2), cv::Scalar(0, 0, 255),
           thickness);
}

int main(int argc, char **argv) {
  // createMatrix();
  // readWriteImage();
  // matrixOperations();
  // readWriteImage();
  // matrixOperations();
  // drawingFunsctionAndCoordinate();
  // getVideoFromCam();
  // matStructure();
  // manipulatingImageChannels();
  // scalarValues();
  // accessingMatrixElements();
  // matrixFromVectorOfCVPoints();
  // displayingVideo();
  // getVideoFromCam() ;
  writingVideo();
}

using namespace cv;
using namespace std;
int main2(int, char **) {
  Mat src;
  // use default camera as video source
  VideoCapture cap(0);
  // check if we succeeded
  if (!cap.isOpened()) {
    cerr << "ERROR! Unable to open camera\n";
    return -1;
  }
  // get one frame from camera to know frame size and type
  cap >> src;
  // check if we succeeded
  if (src.empty()) {
    cerr << "ERROR! blank frame grabbed\n";
    return -1;
  }
  bool isColor = (src.type() == CV_8UC3);
  //--- INITIALIZE VIDEOWRITER
  VideoWriter writer;
  int codec = VideoWriter::fourcc(
      'M', 'J', 'P',
      'G');          // select desired codec (must be available at runtime)
  double fps = 25.0; // framerate of the created video stream
  string filename = "./live.avi"; // name of the output video file
  writer.open(filename, codec, fps, src.size(), isColor);
  // check if we succeeded
  if (!writer.isOpened()) {
    cerr << "Could not open the output video file for write\n";
    return -1;
  }
  //--- GRAB AND WRITE LOOP
  cout << "Writing videofile: " << filename << endl
       << "Press any key to terminate" << endl;
  for (;;) {
    // check if we succeeded
    if (!cap.read(src)) {
      cerr << "ERROR! blank frame grabbed\n";
      break;
    }
    // encode the frame into the videofile stream
    writer.write(src);
    // show live and wait for a key with timeout long enough to show images
    imshow("Live", src);
    if (waitKey(5) >= 0)
      break;
  }
  // the videofile will be closed and released automatically in VideoWriter
  // destructor
  return 0;
}
