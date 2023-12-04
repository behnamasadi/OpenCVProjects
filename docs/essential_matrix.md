# Essential Matrix

 Essential matrix <img src="https://latex.codecogs.com/svg.image?\mathbf{E}" title="https://latex.codecogs.com/svg.image?\mathbf{E}" /> is a <img src="https://latex.codecogs.com/svg.image?3\times&space;3&space;" title="https://latex.codecogs.com/svg.image?3\times 3 " /> that relates optical center of cameras
<img src="images/Epipolar_geometry.svg" />

We can relate any two frame by a rotation matrix and a translation vector:
<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?O_L=RO_R&plus;T&space;" title="https://latex.codecogs.com/svg.image?O_L=RO_R+T " />
<br/>
<br/>

We cross product both side by <img src="https://latex.codecogs.com/svg.image?T" title="https://latex.codecogs.com/svg.image?T" />:
<br/>
<br/>

<img src="https://latex.codecogs.com/svg.image?T\times&space;O_L=T\times&space;RO_R&plus;&space;T\times&space;T&space;" title="https://latex.codecogs.com/svg.image?T\times O_L=T\times RO_R+ T\times T " />

<br/>
<br/>
For any vector cross product of with itself is zero:
<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?T&space;\times&space;T=0" title="https://latex.codecogs.com/svg.image?T \times T=0" />
<br/>
<br/>

We dot product both side by <img src="https://latex.codecogs.com/svg.latex?O_%7BL_%7B1%5Ctimes%203%7D%7D%5ET" title="https://latex.codecogs.com/svg.image?O_{L_{1\times 3}}^T" />:
<br/>
<br/>
<img src="https://latex.codecogs.com/svg.latex?O_%7BL%7D%5ET.%28T%5Ctimes%20O_L%29%3DO_%7BL%7D%5ET.%28T%5Ctimes%20RO_R%29" title="https://latex.codecogs.com/svg.image?O_{L}^T.(T\times O_L)=O_{L}^T.(T\times RO_R) " />
<br/>
<br/>
on the lhs, <img src="https://latex.codecogs.com/svg.image?O_L" title="https://latex.codecogs.com/svg.image?O_L" />   is perpendicular to <img src="https://latex.codecogs.com/svg.image?T\times&space;O_L" title="https://latex.codecogs.com/svg.image?T\times O_L" />
so the result is zero vector, also we can write any cross product as skew-symmetric matrix multiplication, therefore:
<br/>
<br/>


<img src="https://latex.codecogs.com/svg.latex?0_%7B1%20%5Ctimes%201%7D%3DO_%7BL_%7B1%5Ctimes%203%7D%7D%5ET%20%5Cleft%20%5Clfloor%20T%20%5Cright%20%5Crfloor_%7B3%5Ctimes%203%7D%20R_%7B3%5Ctimes%203%7D%20O_%7BR_%7B3%20%5Ctimes%201%7D%20%7D" title="https://latex.codecogs.com/svg.image?0_{1 \times 1}=O_{L_{1\times 3}}^T \left \lfloor T \right \rfloor_{3\times 3} R_{3\times 3} O_{R_{3 \times 1} }" />


<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?0=O_L^T&space;\mathbf{E}&space;O_R&space;&space;" title="https://latex.codecogs.com/svg.image?0=O_L^T \mathbf{E} O_R " />


<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?\mathbf{E}=\left&space;\lfloor&space;T&space;\right&space;\rfloor_{3\times&space;3}&space;&space;&space;&space;&space;&space;R_{3\times&space;3}" title="https://latex.codecogs.com/svg.image?\mathbf{E}=\left \lfloor T \right \rfloor_{3\times 3} R_{3\times 3}" />

# 8-point Algorithm

## Minimal solution
8 point correspondences are needed

The 8-point algorithm assumes that the entries of E are all independent
(which is not true since, for the calibrated case, they depend on 5 parameters (R and T))

The solution of the 8-point algorithm is degenerate when the 3D points are coplanar.
 Conversely, the 5-point algorithm works also for coplanar points

## Over-determined solution
n > 8 points 



                                                              
# 5-point Algorithm

The 5-point algorithm uses the epipolar constraint considering the dependencies among all entries.


Refs: [1](https://en.wikipedia.org/wiki/Essential_matrix)





# Decompose Essential Matrix
4 possible solutions of R and T"



<img src="https://latex.codecogs.com/svg.latex?%5BR_1%2Ct%5D%2C%20%5BR_1%2C-t%5D%2C%20%5BR_2%2Ct%5D%2C%20%5BR_2%2C-t%5D." alt="https://latex.codecogs.com/svg.latex? [R_1,t], [R_1,-t], [R_2,t], [R_2,-t]." />


There exists only one solution where points are in front of both cameras.
(For detail please read chapter 9.6.2 Extraction of cameras from the essential matrix, Multiple View Geometry in Computer Vision (Second Edition))

Refs: [1](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga54a2f5b3f8aeaf6c76d4a31dece85d5d)

<!---
@incollection{grandstrand:2004,
  author      = "Ove Grandstrand",
  title       = "Innovation and Intellectual Property Rights",
  editor      = "Jan Fagerberg and David C. Mowery and Richard R. Nelson",
  booktitle   = "The Oxford Handbook of Innovation",
  publisher   = "Oxford University Press",
  address     = "Oxford",
  year        = 2004,
  pages       = "266-290",
  chapter     = 10,
}
-->



# Properties of the essential matrix


<img src="https://latex.codecogs.com/svg.latex?%5Cmathbf%7BE%7D%3D%5Cleft%20%5Clfloor%20T%20%5Cright%20%5Crfloor_%7B3%5Ctimes%203%7D%20R_%7B3%5Ctimes%203%7D%3DSR" title="https://latex.codecogs.com/svg.image?\mathbf{E}=\left \lfloor T \right \rfloor_{3\times 3} R_{3\times 3}=SR" />

<br/>
<br/>

where `S` is skew-symmetric matrix.

<br/>
<br/>

<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20%5Cmathbf%7BF%7D%3D%5Cmathbf%7BC%7D%5E%7B-T%7D_L%20%5Cmathbf%7BE%7D%20%5Cmathbf%7BC%7D%5E%7B-1%7D_R%20%5C%5C%20%5Cmathbf%7BE%7D%3D%5Cmathbf%7BC%7D%5E%7BT%7D_L%20%5Cmathbf%7BF%7D%20%5Cmathbf%7BC%7D_R" title="https://latex.codecogs.com/svg.image?\\ \mathbf{F}=\mathbf{C}^{-T}_L \mathbf{E} \mathbf{C}^{-1}_R \\ \mathbf{E}=\mathbf{C}^{T}_L \mathbf{F} \mathbf{C}_R" />

<br/>
<br/>

It is possible to bring every skew-symmetric matrix to a block diagonal form by a special orthogonal transformation:


<img src="https://latex.codecogs.com/svg.latex?A%3DQ \Sigma Q%5ET" alt="https://latex.codecogs.com/svg.latex?A=Q\Sigma Q^T" />

<br/>
<br/>


<img src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20Q%5E%7B%5Cmathrm%20%7BT%7D%20%7DQ%3DQQ%5E%7B%5Cmathrm%20%7BT%7D%20%7D%3DI%2C%7D" alt="https://latex.codecogs.com/svg.latex?{\displaystyle Q^{\mathrm {T} }Q=QQ^{\mathrm {T} }=I,}" />

<br/>
<br/>


where <img src="https://latex.codecogs.com/svg.latex?\Sigma" alt="https://latex.codecogs.com/svg.latex?\Sigma" /> is a **block-diagonal** matrix



<img src="https://latex.codecogs.com/svg.latex?%7B%5Cdisplaystyle%20%5CSigma%20%3D%7B%5Cbegin%7Bbmatrix%7D%7B%5Cbegin%7Bmatrix%7D0%26%5Clambda%20_%7B1%7D%5C%5C-%5Clambda%20_%7B1%7D%260%5Cend%7Bmatrix%7D%7D%260%26%5Ccdots%20%260%5C%5C0%26%7B%5Cbegin%7Bmatrix%7D0%26%5Clambda%20_%7B2%7D%5C%5C-%5Clambda%20_%7B2%7D%260%5Cend%7Bmatrix%7D%7D%26%260%5C%5C%5Cvdots%20%26%26%5Cddots%20%26%5Cvdots%20%5C%5C0%260%26%5Ccdots%20%26%7B%5Cbegin%7Bmatrix%7D0%26%5Clambda%20_%7Br%7D%5C%5C-%5Clambda%20_%7Br%7D%260%5Cend%7Bmatrix%7D%7D%5C%5C%26%26%26%26%7B%5Cbegin%7Bmatrix%7D0%5C%5C%26%5Cddots%20%5C%5C%26%260%5Cend%7Bmatrix%7D%7D%5Cend%7Bbmatrix%7D%7D%7D" alt="https://latex.codecogs.com/svg.latex?{\displaystyle \Sigma ={\begin{bmatrix}{\begin{matrix}0&\lambda _{1}\\-\lambda _{1}&0\end{matrix}}&0&\cdots &0\\0&{\begin{matrix}0&\lambda _{2}\\-\lambda _{2}&0\end{matrix}}&&0\\\vdots &&\ddots &\vdots \\0&0&\cdots &{\begin{matrix}0&\lambda _{r}\\-\lambda _{r}&0\end{matrix}}\\&&&&{\begin{matrix}0\\&\ddots \\&&0\end{matrix}}\end{bmatrix}}}" />


<br/>
<br/>

we can write it as:

<img src="https://latex.codecogs.com/svg.latex?Z%3D%5Cbegin%7Bbmatrix%7D%200%20%26%201%20%5C%5C%20-1%20%26%200%5C%5C%20%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?Z=\begin{bmatrix}
0 & 1 \\ 
-1 & 0\\ 
\end{bmatrix}" />




<img src="https://latex.codecogs.com/svg.latex?\Sigma=diag%28a_1Z%2Ca_2Z%2C...%2Ca_mZ%2C0%2C...%2C0%29" alt="https://latex.codecogs.com/svg.latex?\Sigma=diag(a_1Z,a_2Z,...,a_mZ,0,...,0)" />


<br/>
<br/>

A 3×3 matrix is an essential matrix if and only if two of its singular values are equal, and the third is zero.


<img src="https://latex.codecogs.com/svg.latex?W%3D%5Cbegin%7Bbmatrix%7D%200%20%26%20-1%20%26%200%5C%5C%201%20%26%200%20%26%200%5C%5C%200%20%26%200%20%26%201%20%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?W=\begin{bmatrix}
0 & -1 & 0\\ 
1 & 0 & 0\\ 
0 & 0 & 1
\end{bmatrix}" />


<img src="https://latex.codecogs.com/svg.latex?Z%3D%5Cbegin%7Bbmatrix%7D%200%20%26%201%20%26%200%5C%5C%20-1%20%26%200%20%26%200%5C%5C%200%20%26%200%20%26%200%20%5Cend%7Bbmatrix%7D" alt="https://latex.codecogs.com/svg.latex?Z=\begin{bmatrix}
0 & 1 & 0\\ 
-1 & 0 & 0\\ 
0 & 0 & 0
\end{bmatrix}" />



<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20S%3DkUZU%5ET%20%5C%5C%20Z%3Ddiag%281%2C1%2C0%29W%20%5C%5C%20E%3DSR%3DkUdiag%281%2C1%2C0%29WU%5ETR" alt="https://latex.codecogs.com/svg.latex?S=kUZU^T
\\
Z=diag(1,1,0)W
\\
E=SR=kUdiag(1,1,0)WU^TR" />


<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20E%3DSR%3DUdiag%281%2C1%2C0%29V%5E%7BT%7D%20%5C%5C%20P%3D%5BI%7C0%5D%20%5C%5C%20P_1%3D%5BUWV%5ET%7C&plus;u_3%5D%20%5C%5C%20P_2%3D%5BUWV%5ET%7C-u_3%5D%20%5C%5C%20P_3%3D%5BUW%5ETV%5ET%7C&plus;u_3%5D%20%5C%5C%20P_4%3D%5BUW%5ETV%5ET%7C-u_3%5D" alt="https://latex.codecogs.com/svg.latex?\\
E=SR=Udiag(1,1,0)V^{T}
\\
P=[I|0]
\\
P_1=[UWV^T|+u_3]
\\
P_2=[UWV^T|-u_3]
\\
P_3=[UW^TV^T|+u_3]
\\
P_4=[UW^TV^T|-u_3]" />



Suppose that the SVD of E is

<img src="https://latex.codecogs.com/svg.latex?E%3DUdiag%281%2C1%2C0%29V%5ET" alt="https://latex.codecogs.com/svg.latex?E=Udiag(1,1,0)V^T" />



There are two possible factorizations E=SR



<img src="https://latex.codecogs.com/svg.latex?%5C%5C%20S%3DUZU%5ET%20%5C%5C%20%5C%5C%20R%3D%20%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%20UWV%5ET%20%5C%5C%20or%20%5C%5C%20UW%5ETV%5ET%20%5Cend%7Bmatrix%7D%5Cright." alt="https://latex.codecogs.com/svg.latex?\\
S=UZU^T
\\
\\
R=
\left\{\begin{matrix}
UWV^T
\\ 
or
\\
UW^TV^T
\end{matrix}\right." />



The four solutions are:


<img src="images/essential_four_solutions.jpg" alt="" />
 
point X will be in front of both cameras in one of these four solutions only. Thus, testing
with a single point to determine if it is in front of both cameras is sufficient to decide
between the four different solutions for the camera matrix.



<br/>
<br/>

The essential matrix is the specialization of the fundamental matrix to the case of normalized image coordinates
