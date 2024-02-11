# Stereo Calibration

<table>
<tr>
	<td>
		<img src="images/Left_Image.png" width="75%" height="75%" />
	</td>
	<td>
		<img src="images/Right_Image.png" width="75%" height="75%" />
	</td>
<tr>

</table>






```phyton
retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
    objectPoints, all_imagePoints_cam0, all_imagePoints_cam1, cameraMatrix, distCoeffs, cameraMatrix, distCoeffs, imageSize)
```

[code](../scripts/multi_snapshot_stereo.py)
