Installing OpenCV Python
```
pip install --upgrade pip
pip install opencv-python-headless
pip install opencv-contrib-python
```


Installing matplotlib, pandas, pytransform3d
```
conda install numpy
conda install -c conda-forge matplotlib
conda install -c conda-forge pytransform3d
conda install pandas
```
Create softlink
```
ln -s <$HOME>/workspace/OpenCVProjects/scripts .
ln -s <$HOME>/workspace/OpenCVProjects/images/ .
```

activate environment:
```
conda activate OpenCVProject
```

Details of OpenCV Python

```
pip freeze | grep opencv
pip show opencv-python
```
