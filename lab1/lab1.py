import numpy as np
from skimage import io

#subpunct a

arr = np.zeros((9, 400, 600), dtype=np.uint8)

for idx in range(9):
    image = np.load(f'images/car_{idx}.npy')
    arr[idx] = image

print(arr.shape)

#subpunct b

suma = np.sum(arr, axis = 0, dtype = np.uint32)
print(suma)

#subpunct c & d

sume = np.sum(arr, axis=(1, 2))
print(sume)

idx_max = np.argmax(sume)
print(idx_max)

#subpunct e

mean_image = np.mean(arr, axis = 0, dtype = np.uint32)
print(mean_image)
io.imshow(mean_image.astype(np.uint8))
io.show()

#subpunct f

deviatie = np.std(arr, axis = (1 ,2))
print(deviatie)

#subpunct g

arr_float = arr.astype(np.float32)
mean_per_image = np.mean(arr, axis = (1, 2), dtype = np.uint32)
for idx in range(9):
    norm = (arr_float[idx] - mean_per_image[idx]) / deviatie[idx]
    print(norm)

#subpunct h

for idx in range(9):
    sliced = arr[idx][200:301, 280:401]
    print(sliced)


