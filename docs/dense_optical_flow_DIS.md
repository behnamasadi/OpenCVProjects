# DIS Optical Flow
Stands for Dense Inverse Search.

```python
dis = cv2.DISOpticalFlow_create()

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = dis.calc(prev_frame_gray, frame_gray, None)

    # Visualization of the flow
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow("Optical Flow", bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if not ret:
        break
    cv2.imshow('', frame_gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    prev_frame = frame_gray.copy()
```


1. **Initialization of the HSV Image**:
   ```python
   hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
   ```
   Here, a new image `hsv` is being initialized to have the same dimensions as the computed `flow` image. The image is of type HSV (Hue, Saturation, Value), and hence it has 3 channels. All pixels are initialized to zero.

2. **Setting Saturation to Maximum**:
   ```python
   hsv[..., 1] = 255
   ```
   The saturation channel of the HSV image is set to the maximum value of 255. This is done to make the output color as vibrant as possible, which makes the visualization of the flow vectors more apparent.

3. **Converting Cartesian to Polar**:
   ```python
   mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
   ```
   The `flow` image contains horizontal and vertical displacement values. This can be thought of as Cartesian coordinates. `cv2.cartToPolar` converts these Cartesian coordinates into polar coordinates, providing the magnitude (`mag`) and angle (`ang`) of the flow vectors. 

4. **Setting the Hue Based on Flow Direction**:
   ```python
   hsv[..., 0] = ang * 180 / np.pi / 2
   ```
   The angle (`ang`) is in radians, ranging from \(0\) to \(2\pi\). This line of code converts the angle to degrees (using `180/np.pi`) and then scales it to fit within the range of the Hue channel in the HSV color space, which is typically from 0 to 180.

5. **Setting the Value Based on Flow Magnitude**:
   ```python
   hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
   ```
   The magnitude of the flow is normalized to fit within the range of the Value channel in the HSV color space, which is from 0 to 255. The normalization ensures that small and large flow magnitudes get mapped to the full range of available brightness values.

To summarize:
- The **Hue** channel represents the direction of the motion (angle of the flow vector).
- The **Saturation** channel is kept at maximum to ensure vibrant colors.
- The **Value** channel represents the magnitude of the motion (length of the flow vector).
