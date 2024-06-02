import numpy as np # shorten numpy to np for readability

array = np.array([1, 2, 3, 4, 5, 6])
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Accessing elements of a array using square brackets and indexes
print(arr[0]) # Output -> [1, 2, 3, 4]

arr1 = np.array([[0., 0., 0.], [1., 1., 1.]]) # Array has 2 axes(2D array)

print(arr1.shape)

# Creating array filled with zeros & ones
arr2 = np.ones(3)
print(arr2) # Output -> [1. 1. 1.]
arr3 = np.zeros(5)
print(arr3) # Output -> [0. 0. 0. 0. 0.]
# Create an empty array with 4 elements
print(np.empty(4)) # Random arrays changed afterwards

# create an array with a range of elements
print(np.arange(9)) # Output -> [0 1 2 3 4 5 6 7 8]

# Specifying the start, intervals and stop
print(np.arange(3, 15, 3)) # Output -> [3 6 9 12]

print(np.linspace(0, 10, num=5)) # Output -> [ 0.   2.5  5.   7.5 10. ]

# Specifying data type
print(np.empty(4, dtype=np.int64)) 
print(np.ones(3, dtype=np.int64))

array1 = np.array([23, 8, 12, 42, 89, 12, 99, 8])
print(np.sort(array1))

a = np.array([2, 89, 9, 2])
b = np.array([9, 8, 23, 20])
print(np.concatenate((a, b)))

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6]])
print(np.concatenate((x, y), axis=0))

arr4 = np.array([
  [[1, 2], 
   [3, 4],
   [3, 4]],

  [[5, 6], 
   [7, 8],
   [3, 4]]
]
)

print(arr4.shape) # Output -> (2, 3, 2)
print(arr4.ndim) # Output -> 3-D
print(arr4.size) # Output -> 2 * 3 * 2 = 12

# Remember while reshaping the sizes remain constant
print(arr4.reshape(2, 6)) # Output -> [[1 2 3 4 3 4] [5 6 7 8 3 4]]

print(array)
print(array.shape)

array2 = array[np.newaxis, :]
print(array2)
print(array2.shape)

col_vector = array[:, np.newaxis]
print(col_vector)
print(col_vector.shape)
