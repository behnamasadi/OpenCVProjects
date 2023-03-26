# Fundamental Matrix

If we have camera matrix we can compute the **ray** that goes the through optical center of <img src="https://latex.codecogs.com/svg.image?O_L" title="https://latex.codecogs.com/svg.image?O_L" /> (left cameras) and <img src="https://latex.codecogs.com/svg.image?O_R" title="https://latex.codecogs.com/svg.image?O_R" /> (right camera) and the point <img src="https://latex.codecogs.com/svg.image?X" title="https://latex.codecogs.com/svg.image?X" /> :


<img src="images/TriangulationIdeal.svg" />  

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?\mathbf{y}_1=&space;\begin{bmatrix}y_1{_u}&space;\\y_1_{v}&space;\\1\end{bmatrix}," title="https://latex.codecogs.com/svg.image?\mathbf{y}_1= \begin{bmatrix}y_1{_u} \\y_1_{v} \\1\end{bmatrix}," />

<img src="https://latex.codecogs.com/svg.image?\mathbf{y}_2=&space;\begin{bmatrix}y_2{_u}&space;\\y_2_{v}&space;\\1\end{bmatrix}" title="https://latex.codecogs.com/svg.image?\mathbf{y}_2= \begin{bmatrix}y_2{_u} \\y_2_{v} \\1\end{bmatrix}" />


<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?ray_1=\mathbf{C}^{-1}_1{_{3\times3}}&space;\times&space;\begin{bmatrix}y_1{_u}&space;\\y_1_{v}&space;\\1\end{bmatrix}=\mathbf{C}^{-1}_1\mathbf{y}_1" title="https://latex.codecogs.com/svg.image?ray_1=\mathbf{C}^{-1}_1{_{3\times3}} \times \begin{bmatrix}y_1{_u} \\y_1_{v} \\1\end{bmatrix}=\mathbf{C}^{-1}_1\mathbf{y}_1" />
<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?ray_2=\mathbf{C}^{-1}_2{_{3\times3}}&space;\times&space;\begin{bmatrix}y_2{_u}&space;\\y_2_{v}&space;\\1\end{bmatrix}=\mathbf{C}^{-1}_2\mathbf{y}_2" title="https://latex.codecogs.com/svg.image?ray_2=\mathbf{C}^{-1}_2{_{3\times3}} \times \begin{bmatrix}y_2{_u} \\y_2_{v} \\1\end{bmatrix}=\mathbf{C}^{-1}_2\mathbf{y}_2" />


<br/>
<br/>





Also from essential matrix we know:
<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?0=O_L^T&space;\mathbf{E}&space;O_R&space;&space;" title="https://latex.codecogs.com/svg.image?0=O_L^T \mathbf{E} O_R " />


<br/>
<br/>
if we substitute them in the upper equation:
<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?0=(\mathbf{C}^{-1}_L\mathbf{y}_L)^T&space;\mathbf{E}&space;\mathbf{C}^{-1}_R\mathbf{y}_R" title="https://latex.codecogs.com/svg.image?0=(\mathbf{C}^{-1}_L\mathbf{y}_L)^T \mathbf{E} \mathbf{C}^{-1}_R\mathbf{y}_R" />

<br/>
<br/>


<img src="https://latex.codecogs.com/svg.image?0=\mathbf{y}_L&space;^T\mathbf{C}^{-T}_L&space;\mathbf{E}&space;\mathbf{C}^{-1}_R\mathbf{y}_R" title="https://latex.codecogs.com/svg.image?0=\mathbf{y}_L ^T\mathbf{C}^{-T}_L \mathbf{E} \mathbf{C}^{-1}_R\mathbf{y}_R" />
<br/>
<br/>

we call the <img src="https://latex.codecogs.com/svg.latex?%5Cmathbf%7BC%7D%5E%7B-T%7D_L%20%5Cmathbf%7BE%7D%20%5Cmathbf%7BC%7D%5E%7B-1%7D_R" title="https://latex.codecogs.com/svg.image?\mathbf{C}^{-T}_L \mathbf{E} \mathbf{C}^{-1}_R" /> matrix **Fundamental Matrix**: 

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20%5Cmathbf%7BF%7D%3D%5Cmathbf%7BC%7D%5E%7B-T%7D_L%20%5Cmathbf%7BE%7D%20%5Cmathbf%7BC%7D%5E%7B-1%7D_R%20%5C%5C%20%5Cmathbf%7BE%7D%3D%5Cmathbf%7BC%7D%5E%7BT%7D_L%20%5Cmathbf%7BF%7D%20%5Cmathbf%7BC%7D_R" title="https://latex.codecogs.com/svg.image?\\ \mathbf{F}=\mathbf{C}^{-T}_L \mathbf{E} \mathbf{C}^{-1}_R \\ \mathbf{E}=\mathbf{C}^{T}_L \mathbf{F} \mathbf{C}_R" />

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?0=\mathbf{y}_L&space;^T\mathbf{F}&space;\mathbf{y}_R" title="https://latex.codecogs.com/svg.image?0=\mathbf{y}_L ^T\mathbf{F} \mathbf{y}_R" />
<br/>
<br/>


Now how to compute it:
<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}u&space;&&space;v&space;&&space;1&space;\\\end{bmatrix}\begin{bmatrix}f_{11}&space;&&space;f_{12}&space;&&space;f_{13}&space;\\f_{21}&space;&&space;f_{22}&space;&&space;f_{23}&space;\\f_{31}&space;&&space;f_{32}&space;&&space;f_{33}&space;\\\end{bmatrix}\begin{bmatrix}u'&space;\\v'&space;\\1\end{bmatrix}=0" title="https://latex.codecogs.com/svg.image?\begin{bmatrix}u & v & 1 \\\end{bmatrix}\begin{bmatrix}f_{11} & f_{12} & f_{13} \\f_{21} & f_{22} & f_{23} \\f_{31} & f_{32} & f_{33} \\\end{bmatrix}\begin{bmatrix}u' \\v' \\1\end{bmatrix}=0" />

if we write this equation for 8 points, we can arrange the following matrix:


<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}u'_1u_1&space;&&space;u'_1v_1&space;&&space;u'_1&space;&&space;v'_1u_1&space;&&space;v'_1v_1&space;&&space;v'_1&space;&&space;u_1&space;&&space;v_1&space;&&space;1&space;\\u'_2u_2&space;&&space;u'_2v_2&space;&&space;u'_2&space;&&space;v'_2u_2&space;&&space;v'_2v_2&space;&&space;v'_2&space;&&space;u_2&space;&&space;v_2&space;&&space;1&space;\\u'_3u_3&space;&&space;u'_3v_3&space;&&space;u'_3&space;&&space;v'_3u_3&space;&&space;v'_3v_3&space;&&space;v'_3&space;&&space;u_3&space;&&space;v_3&space;&&space;1&space;\\u'_4u_4&space;&&space;u'_4v_4&space;&&space;u'_4&space;&&space;v'_4u_4&space;&&space;v'_4v_4&space;&&space;v'_4&space;&&space;u_4&space;&&space;v_4&space;&&space;1&space;\\u'_5u_5&space;&&space;u'_5v_5&space;&&space;u'_5&space;&&space;v'_5u_5&space;&&space;v'_5v_5&space;&&space;v'_5&space;&&space;u_5&space;&&space;v_5&space;&&space;1&space;\\u'_6u_6&space;&&space;u'_6v_6&space;&&space;u'_6&space;&&space;v'_6u_6&space;&&space;v'_6v_6&space;&&space;v'_6&space;&&space;u_6&space;&&space;v_6&space;&&space;1&space;\\u'_7u_7&space;&&space;u'_7v_7&space;&&space;u'_7&space;&&space;v'_7u_7&space;&&space;v'_7v_7&space;&&space;v'_7&space;&&space;u_7&space;&&space;v_7&space;&&space;1&space;\\u'_8u_8&space;&&space;u'_8v_8&space;&&space;u'_8&space;&&space;v'_8u_8&space;&&space;v'_8v_8&space;&&space;v'_8&space;&&space;u_8&space;&&space;v_8&space;&&space;1&space;\\\end{bmatrix}\begin{bmatrix}f_{11}&space;\\&space;f_{12}&space;\\&space;f_{13}&space;\\f_{21}&space;\\&space;f_{22}&space;\\&space;f_{23}&space;\\f_{31}&space;\\&space;f_{32}&space;\\&space;f_{33}&space;\\\end{bmatrix}=0" title="https://latex.codecogs.com/svg.image?\begin{bmatrix}u'_1u_1 & u'_1v_1 & u'_1 & v'_1u_1 & v'_1v_1 & v'_1 & u_1 & v_1 & 1 \\u'_2u_2 & u'_2v_2 & u'_2 & v'_2u_2 & v'_2v_2 & v'_2 & u_2 & v_2 & 1 \\u'_3u_3 & u'_3v_3 & u'_3 & v'_3u_3 & v'_3v_3 & v'_3 & u_3 & v_3 & 1 \\u'_4u_4 & u'_4v_4 & u'_4 & v'_4u_4 & v'_4v_4 & v'_4 & u_4 & v_4 & 1 \\u'_5u_5 & u'_5v_5 & u'_5 & v'_5u_5 & v'_5v_5 & v'_5 & u_5 & v_5 & 1 \\u'_6u_6 & u'_6v_6 & u'_6 & v'_6u_6 & v'_6v_6 & v'_6 & u_6 & v_6 & 1 \\u'_7u_7 & u'_7v_7 & u'_7 & v'_7u_7 & v'_7v_7 & v'_7 & u_7 & v_7 & 1 \\u'_8u_8 & u'_8v_8 & u'_8 & v'_8u_8 & v'_8v_8 & v'_8 & u_8 & v_8 & 1 \\\end{bmatrix}\begin{bmatrix}f_{11} \\ f_{12} \\ f_{13} \\f_{21} \\ f_{22} \\ f_{23} \\f_{31} \\ f_{32} \\ f_{33} \\\end{bmatrix}=0" />
<br/>
<br/>
the above equation has zero on the rhs, so we can use SVD to solve it,
<br/>
<br/>


<img src="https://latex.codecogs.com/svg.image?F^{*}%20\underset{H}{\mathrm{argmin}}=%20\|AF\|^{2}" title="https://latex.codecogs.com/svg.image?F^{*} \underset{H}{\mathrm{argmin}}= \|AF\|^{2}" />




Singular-value Decomposition (SVD) of any given matrix <img src="https://latex.codecogs.com/svg.image?A_{M{\times}N}" title="https://latex.codecogs.com/svg.image?A_{M{\times}N}" />

<br/>
<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?\begin{equation}&space;\underbrace{\mathbf{A}}_{M&space;\times&space;N}&space;=&space;\underbrace{\mathbf{U}}_{M&space;\times&space;M}&space;\times&space;\underbrace{\mathbf{\Sigma}}_{M\times&space;N}&space;\times&space;\underbrace{\mathbf{V}^{\text{T}}}_{N&space;\times&space;N}&space;\end{equation}" title="https://latex.codecogs.com/svg.image?\begin{equation} \underbrace{\mathbf{A}}_{M \times N} = \underbrace{\mathbf{U}}_{M \times M} \times \underbrace{\mathbf{\Sigma}}_{M\times N} \times \underbrace{\mathbf{V}^{\text{T}}}_{N \times N} \end{equation}" />




- <img src="https://latex.codecogs.com/svg.image?U" title="https://latex.codecogs.com/svg.image?U" /> is an <img src="https://latex.codecogs.com/svg.image?M\times&space;M" title="https://latex.codecogs.com/svg.image?M\times M" /> matrix with orthogonal matrix (columns are eigen vectors of A).
- <img src="https://latex.codecogs.com/svg.image?\Sigma" title="https://latex.codecogs.com/svg.image?\Sigma" /> is an <img src="https://latex.codecogs.com/svg.image?M\times&space;N" title="https://latex.codecogs.com/svg.image?M\times N" /> matrix with non-negative entries, termed the singular values  (diagonal entries are eigen values of A).
- <img src="https://latex.codecogs.com/svg.image?V" title="https://latex.codecogs.com/svg.image?V" /> is an <img src="https://latex.codecogs.com/svg.image?N\times&space;N" title="https://latex.codecogs.com/svg.image?N\times N" /> orthogonal matrix.




<img src="https://latex.codecogs.com/svg.image?F^{*}" title="https://latex.codecogs.com/svg.image?F^{*}" /> is the last column of <img src="https://latex.codecogs.com/svg.image?V" title="https://latex.codecogs.com/svg.image?V" />


However, this is not the complete answer, let have a review: the cross product can be written as:
<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?\mathbf&space;{a}&space;\times&space;\mathbf&space;{b}&space;=[\mathbf&space;{a}&space;]_{\times&space;}\mathbf&space;{b}" title="https://latex.codecogs.com/svg.image?\mathbf {a} \times \mathbf {b} =[\mathbf {a} ]_{\times }\mathbf {b}" />


where:
<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?[\mathbf&space;{a}&space;]_{\times&space;}={\begin{bmatrix}\,\,0&\!-a_{3}&\,\,\,a_{2}\\\,\,\,a_{3}&0&\!-a_{1}\\\!-a_{2}&\,\,a_{1}&\,\,0\end{bmatrix}}" title="https://latex.codecogs.com/svg.image?[\mathbf {a} ]_{\times }={\begin{bmatrix}\,\,0&\!-a_{3}&\,\,\,a_{2}\\\,\,\,a_{3}&0&\!-a_{1}\\\!-a_{2}&\,\,a_{1}&\,\,0\end{bmatrix}}" />

This matrix has rank of 2 and if we multiply it by any matrix, that would also have rank 2.


<img src="https://latex.codecogs.com/svg.image?\Sigma=\begin{bmatrix}&space;r&&space;0&space;&&space;0&space;\\&space;0&&space;&space;s&&space;&space;0\\&space;0&&space;&space;0&&space;t&space;\\\end{bmatrix}" title="https://latex.codecogs.com/svg.image?\Sigma=\begin{bmatrix} r& 0 & 0 \\ 0& s& 0\\ 0& 0& t \\\end{bmatrix}" />
<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?r>s>t" title="https://latex.codecogs.com/svg.image?r>s>t" />
<br/>
<br/>
so we set the <img src="https://latex.codecogs.com/svg.image?t=0" title="https://latex.codecogs.com/svg.image?t=0" /> is 
<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?\hat{\Sigma}&space;=\begin{bmatrix}&space;r&&space;0&space;&&space;0&space;\\&space;0&&space;&space;s&&space;&space;0\\&space;0&&space;&space;0&&space;0&space;\\\end{bmatrix}" title="https://latex.codecogs.com/svg.image?\hat{\Sigma} =\begin{bmatrix} r& 0 & 0 \\ 0& s& 0\\ 0& 0& 0 \\\end{bmatrix}" />


and compute <img src="https://latex.codecogs.com/svg.image?\hat{A}" title="https://latex.codecogs.com/svg.image?\hat{A}" />:
<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?\hat{A}=U\times&space;\hat{\Sigma}&space;&space;\times&space;V^T" title="https://latex.codecogs.com/svg.image?\hat{A}=U\times \hat{\Sigma} \times V^T" />


If we compute the SVD of <img src="https://latex.codecogs.com/svg.image?\hat{A}" title="https://latex.codecogs.com/svg.image?\hat{A}" /> the last column of <img src="https://latex.codecogs.com/svg.image?V" title="https://latex.codecogs.com/svg.image?V" /> is <img src="https://latex.codecogs.com/svg.image?F^{*}" title="https://latex.codecogs.com/svg.image?F^{*}" /> that we looking for.

Now that we have <img src="https://latex.codecogs.com/svg.image?F" title="https://latex.codecogs.com/svg.image?F" /> if we set a point on the left camera <img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}&space;u&&space;v&space;&&space;1&space;\\\end{bmatrix}" title="https://latex.codecogs.com/svg.image?\begin{bmatrix} u& v & 1 \\\end{bmatrix}" /> it will give us a line on the other camera.
<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?\begin{bmatrix}u&space;&&space;v&space;&&space;1&space;\\\end{bmatrix}\begin{bmatrix}f_{11}&space;&&space;f_{12}&space;&&space;f_{13}&space;\\f_{21}&space;&&space;f_{22}&space;&&space;f_{23}&space;\\f_{31}&space;&&space;f_{32}&space;&&space;f_{33}&space;\\\end{bmatrix}\begin{bmatrix}u'&space;\\v'&space;\\1\end{bmatrix}=0" title="https://latex.codecogs.com/svg.image?\begin{bmatrix}u & v & 1 \\\end{bmatrix}\begin{bmatrix}f_{11} & f_{12} & f_{13} \\f_{21} & f_{22} & f_{23} \\f_{31} & f_{32} & f_{33} \\\end{bmatrix}\begin{bmatrix}u' \\v' \\1\end{bmatrix}=0" />


To find the epipoles, since every line should go through that, so in the following equation:

<img src="https://latex.codecogs.com/svg.image?0=\mathbf{y}_L&space;^T\mathbf{F}&space;\mathbf{y}_R" title="https://latex.codecogs.com/svg.image?0=\mathbf{y}_L ^T\mathbf{F} \mathbf{y}_R" />
<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?\mathbf{F}&space;\mathbf{y}_R=0" title="https://latex.codecogs.com/svg.image?\mathbf{F} \mathbf{y}_R=0" />
<br/>
<br/>




[code](../src/fundamental_matrix_estimation.cpp)    



Refs: [1](https://www8.cs.umu.se/kurser/TDBD19/VT05/reconstruct-4.pdf), [2](http://www.robots.ox.ac.uk/~vgg/hzbook/code/)


# Normalized 8-point Algorithm
Problem with 8-point algorithm: Poor numerical conditioning, which makes results very sensitive to noise and can be fixed by rescaling the data.
Idea: Transform image coordinates so that they are in the range: 




The Normalized 8-point algorithm can be summarized in three steps:

1. Normalize the point correspondences:


<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20%5Chat%7Bp_1%7D%3DB_1p_1%20%5C%5C%20%5Chat%7Bp_2%7D%3DB_2p_2" alt="https://latex.codecogs.com/svg.latex?\\
\hat{p_1}=B_1p_1 
\\
\hat{p_2}=B_2p_2"/>

<br/>
<br/>
<img src="https://latex.codecogs.com/svg.latex?%5Chat%7Bp_i%7D%3D%5Cfrac%7B%5Csqrt%7B2%7D%7D%7B%5Csigma%7D%20%28p_i-%5Cmu%29" alt="https://latex.codecogs.com/svg.latex?\hat{p_i}=\frac{\sqrt{2}}{\sigma} (p_i-\mu) " />
<br/>
<br/>


<img src="https://latex.codecogs.com/svg.latex?%5Cmu%3D%5Cbegin%7Bpmatrix%7D%20%5Cmu_x%5C%5C%20%5Cmu_y%20%5Cend%7Bpmatrix%7D%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20p_i" alt="https://latex.codecogs.com/svg.latex?"  />
<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?%5Csigma%3D%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5E%7BN%7D%20%7C%7Cp_i-%5Cmu%7C%7C%5E2" alt="https://latex.codecogs.com/svg.latex?\sigma=\frac{1}{N}\sum_{i=1}^{N} ||p_i-\mu||^2"  />


<br/>
<br/>


<img src="https://latex.codecogs.com/svg.latex?%5Chat%7Bp_i%7D%3D%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B%5Csqrt%7B2%7D%7D%7B%5Csigma%7D%20%26%200%20%26%20-%5Cfrac%7B%5Csqrt%7B2%7D%7D%7B%5Csigma%7D%20%5Cmu_x%5C%5C%200%20%26%20%5Cfrac%7B%5Csqrt%7B2%7D%7D%7B%5Csigma%7D%20%26%20-%5Cfrac%7B%5Csqrt%7B2%7D%7D%7B%5Csigma%7D%5Cmu_y%20%5C%5C%200%20%26%200%20%26%201%20%5Cend%7Bbmatrix%7Dp_i" alt="https://latex.codecogs.com/svg.latex?\hat{p_i}=\begin{bmatrix}
\frac{\sqrt{2}}{\sigma}  & 0 & -\frac{\sqrt{2}}{\sigma} \mu_x\\ 
0 & \frac{\sqrt{2}}{\sigma}  & -\frac{\sqrt{2}}{\sigma}\mu_y \\ 
0 & 0 & 1
\end{bmatrix}p_i 
" />




<img src="" alt="" />
<img src="" alt="" />



2. Estimate normalized <img src="https://latex.codecogs.com/svg.latex?%5Chat%7BF%7D" alt="https://latex.codecogs.com/svg.latex?\hat{F}" /> with 8-point algorithm using normalized coordinates

3. Compute unnormalized <img src="https://latex.codecogs.com/svg.latex?F" alt="https://latex.codecogs.com/svg.latex?F" /> from <img src="https://latex.codecogs.com/svg.latex?%5Chat%7BF%7D" alt="https://latex.codecogs.com/svg.latex?\hat{F}" />:


<img src="https://latex.codecogs.com/svg.latex?F%3DB_2%5ET%5Chat%7BF%7DB_1" alt="https://latex.codecogs.com/svg.latex?F=B_2^T\hat{F}B_1" />

# Extract Translation and Rotation from Fundamental Matrix

if the coordinates of the principal points of each camera are known
and the two cameras have the same focal length 𝑓 in pixels, then 𝑅, 𝑇, 𝑓 can
determined uniquely


Refs: [1](https://www.cse.unr.edu/~bebis/CS485/Handouts/hartley.pdf)
