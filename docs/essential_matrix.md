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
For any vector cross product of with itselt is zero:
<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?T&space;\times&space;T=0" title="https://latex.codecogs.com/svg.image?T \times T=0" />
<br/>
<br/>

We dot product both side by <img src="https://latex.codecogs.com/svg.image?O_L" title="https://latex.codecogs.com/svg.image?O_L" />:
<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?O_L.(T\times&space;O_L)=O_L.(T\times&space;RO_R)&space;" title="https://latex.codecogs.com/svg.image?O_L.(T\times O_L)=O_L.(T\times RO_R) " />
<br/>
<br/>
on the lhs, <img src="https://latex.codecogs.com/svg.image?O_L" title="https://latex.codecogs.com/svg.image?O_L" />   is perpendicular to <img src="https://latex.codecogs.com/svg.image?T\times&space;O_L" title="https://latex.codecogs.com/svg.image?T\times O_L" />
so the result is zero vector, also we can write any cross product as skew-symmetric matrix multiplication, therefore:
<br/>
<br/>


<img src="https://latex.codecogs.com/svg.image?0=O_L^T&space;&space;\left&space;\lfloor&space;T&space;\right&space;\rfloor_{3\times&space;3}&space;&space;&space;&space;&space;&space;R_{3\times&space;3}&space;O_R&space;" title="https://latex.codecogs.com/svg.image?0=O_L^T \left \lfloor T \right \rfloor_{3\times 3} R_{3\times 3} O_R " />


<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?0=O_L^T&space;\mathbf{E}&space;O_R&space;&space;" title="https://latex.codecogs.com/svg.image?0=O_L^T \mathbf{E} O_R " />


<br/>
<br/>
<img src="https://latex.codecogs.com/svg.image?\mathbf{E}=\left&space;\lfloor&space;T&space;\right&space;\rfloor_{3\times&space;3}&space;&space;&space;&space;&space;&space;R_{3\times&space;3}" title="https://latex.codecogs.com/svg.image?\mathbf{E}=\left \lfloor T \right \rfloor_{3\times 3} R_{3\times 3}" />


Refs: [1](https://en.wikipedia.org/wiki/Essential_matrix)



# Decompose Essential Matrix

Refs: [1](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga54a2f5b3f8aeaf6c76d4a31dece85d5d),


/*
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
*/

9.6.2 Extraction of cameras from the essential matrix



