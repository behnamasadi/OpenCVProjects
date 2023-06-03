## Image registration
Image registration is the process of transforming different sets of data into one coordinate system. The traditional image registration 
problem can be characterized as follows: Given two functions: <img src="https://latex.codecogs.com/svg.latex?F%28x%29" alt="https://latex.codecogs.com/svg.latex?F(x)" />  and <img src="https://latex.codecogs.com/svg.latex?G%28x%29" alt="https://latex.codecogs.com/svg.latex?G(x)" /> representing pixel values at each location <img src="https://latex.codecogs.com/svg.latex?x" alt="https://latex.codecogs.com/svg.latex?x" /> in two images, respectively, where <img src="https://latex.codecogs.com/svg.latex?x" alt="https://latex.codecogs.com/svg.latex?x" /> is a vector. We wish to find the disparity vector <img src="https://latex.codecogs.com/svg.latex?h" alt="https://latex.codecogs.com/svg.latex?h" /> that minimizes some measure of the difference between <img src="https://latex.codecogs.com/svg.latex?F%28x&plus;h%29" alt="https://latex.codecogs.com/svg.latex?F(x+h)" />  and <img src="https://latex.codecogs.com/svg.latex?G%28x%29" alt="https://latex.codecogs.com/svg.latex?G(x)" />, for <img src="https://latex.codecogs.com/svg.latex?x" alt="https://latex.codecogs.com/svg.latex?x" /> in some region of interest 
<img src="https://latex.codecogs.com/svg.latex?R" alt="https://latex.codecogs.com/svg.latex?R" />



## Kanade–Lucas–Tomasi feature tracker

The basic idea behind the KLT tracker is to track a set of carefully selected feature points across consecutive frames of a video. These feature points can be corners, edges, or other distinctive parts of the object or scene being tracked. The tracker assumes that the position of these feature points does not change significantly between adjacent frames, except for small translations and deformations.
