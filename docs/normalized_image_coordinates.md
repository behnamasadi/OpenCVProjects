# Normalized Image Coordinates

Consider a camera matrix decomposed as <img src="https://latex.codecogs.com/svg.latex?P%20%3D%20K%5BR%20%7C%20t%5D" alt="https://latex.codecogs.com/svg.latex?P = K[R | t]" />,
and let <img src="https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bx%7D%20%3D%20P%20%5Cmathbf%7BX%7D" alt="https://latex.codecogs.com/svg.latex?\mathbf{x} = P \mathbf{X}" /> be a point in the image. If the calibration matrix <img src="https://latex.codecogs.com/svg.latex?K" alt="https://latex.codecogs.com/svg.latex?K" /> is known, then we
may apply its inverse to the point <img src="https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bx%7D" alt="https://latex.codecogs.com/svg.latex?\mathbf{x}" /> to obtain the point <img src="https://latex.codecogs.com/svg.latex?%5Chat%7B%5Cmathbf%7Bx%7D%7D%20%3DK%20%5E%7B-1%7D%20%5Cmathbf%7Bx%7D" alt="https://latex.codecogs.com/svg.latex?\hat{\mathbf{x}} =K ^{-1} \mathbf{x}" />. Then <img src="https://latex.codecogs.com/svg.latex?%5Chat%7B%5Cmathbf%7Bx%7D%7D%20%3D%5BR%7Ct%5D%5Cmathbf%7BX%7D" alt="https://latex.codecogs.com/svg.latex?\hat{\mathbf{x}} =[R|t]\mathbf{X}" />,
where <img src="https://latex.codecogs.com/svg.latex?%5Chat%7B%5Cmathbf%7Bx%7D%7D" alt="https://latex.codecogs.com/svg.latex?\hat{\mathbf{x}}" /> is the image point expressed in normalized coordinates. It may be thought of
as the image of the point <img src="https://latex.codecogs.com/svg.latex?%5Chat%7B%5Cmathbf%7BX%7D%7D" alt="https://latex.codecogs.com/svg.latex?\hat{\mathbf{X}}" /> with respect to a camera <img src="https://latex.codecogs.com/svg.latex?%5BR%7Ct%5D" alt="https://latex.codecogs.com/svg.latex?[R|t]" /> having the identity matrix I
as calibration matrix. The camera matrix <img src="https://latex.codecogs.com/svg.latex?K%5E%7B-1%7DP%3D%5BR%7Ct%5D" alt="https://latex.codecogs.com/svg.latex?K^{-1}P=[R|t]" /> is called a **normalized camera
matrix**, the effect of the known calibration matrix having been removed.
Now, consider a pair of normalized camera matrices <img src="https://latex.codecogs.com/svg.latex?P%20%3D%20%5BI%20%7C%200%20%5D" alt="https://latex.codecogs.com/svg.latex?P = [I | 0 ]" /> and <img src="https://latex.codecogs.com/svg.latex?P%5E%7B%5Cprime%7D%20%3D%20%5BR%20%7C%20t%20%5D" alt="https://latex.codecogs.com/svg.latex?P^{\prime} = [R | t ]" />. The
fundamental matrix corresponding to the pair of normalized cameras is customarily
called the essential matrix, and has the form:

<img src="https://latex.codecogs.com/svg.latex?E%3D%5Br%5D_%7B%5Ctimes%7DR%3DR%5BR%5ETt%5D_%7B%5Ctimes%7D." alt="https://latex.codecogs.com/svg.latex?E=[r]_{\times}R=R[R^Tt]_{\times}." />






<img src="" alt="" />

















normalize your image coordinate in pixel to the range [-1 ; 1]:


To provide a uniform treatment of image coordinates between modules, independently of image size / resolution (different modules may use different image resolutions), it is convenient to adopt the following standard:

Let the image have size (w, h), where w is the image width and h is the image height.
Let the image coordinates be (u,v), in pixel units, where u is the horizontal coordinate, starting at the top-left corner and growing to the right, and v is the vertical image coordinate, starting at the top-left corner and growing to the bottom.
The normalized image coordinates are defined as:
x = 2*u/w - 1

y = 2*v/h - 1

Thus, the normalized image coordinates (both x and y) are in the range [-1,1], being 0 the center of the image, -1 the left/top boundaries, and 1 the right/bottom boundaries.

The x coordinate starts at the image center and grows to the right. The y coordinate starts at the image center and grows to the bottom.

To go back to image pixels, the following expressions should be used:

u = w*(x+1)/2

v = h*(y+1)/2


# Normalized Camera Matrix 

you normalize in order to have the normalized image plane located at the focal length=1 by dividing the 3D coordinate expressed in the camera frame by the Z coordinate (


The camera matrix derived above can be simplified even further if we assume that f = 1:



<img src="https://latex.codecogs.com/svg.latex?%7B%5Cmathbf%20%7BC%7D%7D_%7B%7B0%7D%7D%3D%7B%5Cbegin%7Bpmatrix%7D1%260%260%260%5C%5C0%261%260%260%5C%5C0%260%261%260%5Cend%7Bpmatrix%7D%7D%3D%5Cleft%28%7B%5Cbegin%7Barray%7D%7Bc%7Cc%7D%7B%5Cmathbf%20%7BI%7D%7D%26%7B%5Cmathbf%20%7B0%7D%7D%5Cend%7Barray%7D%7D%5Cright%29" alt="{\mathbf  {C}}_{{0}}={\begin{pmatrix}1&0&0&0\\0&1&0&0\\0&0&1&0\end{pmatrix}}=\left({\begin{array}{c|c}{\mathbf  {I}}&{\mathbf  {0}}\end{array}}\right)"/>

Refs: [1](http://wiki.icub.org/wiki/Image_Coordinate_Standard), [2](https://en.wikipedia.org/wiki/Camera_matrix#Normalized_camera_matrix_and_normalized_image_coordinates), [3](http://users.ece.northwestern.edu/~yingwu/teaching/EECS432/Notes/camera.pdf), [4](https://opensfm.readthedocs.io/en/latest/geometry.html), [5](https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/agartia3/index.html#:~:text=Normalized%20images%20are%20mean%20centred,as%20a%20measure%20of%20performance.), [6](https://www.csc.kth.se/~madry/courses/mvg10/Attachments/Oscar_norm_8pt_alg.pdf), [7](https://robotics.stackexchange.com/questions/12741/normalized-point-coordinates-in-undistortpoints-function)


# General Camera Matrix




