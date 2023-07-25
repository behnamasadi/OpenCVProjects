The reasoning behind this method is that sharp images will have a wide range of edges, hence the variance of the Laplacian (which is an edge detector) will be high. Conversely, blurry images will have fewer edges, resulting in a lower variance of the Laplacian.

```
laplacian = cv2.Laplacian(image, cv2.CV_64F)

print(laplacian)
# Compute the variance
laplacian_variance = np.var(laplacian)

print(laplacian_variance)
return laplacian_variance < threshold
```    
