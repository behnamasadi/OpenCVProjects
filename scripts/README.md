
pip install --upgrade pip
pip install opencv-python-headless
pip install opencv-contrib-python

//conda install -c conda-forge opencv
conda install numpy
conda install -c conda-forge matplotlib
conda install -c conda-forge pytransform3d

ln -s /home/behnam/workspace/OpenCVProjects/scripts .
ln -s /home/behnam/workspace/OpenCVProjects/images/ .


pip freeze | grep opencv
pip show opencv-python

