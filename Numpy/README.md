# NumPy (Numerical Python) Tutorial

## Installing NumPy

```bash
pip install numpy
```

## Difference between Python lists and a NumPy array?

- Lists can contain different data types while all of the elements in NumPy array should be homogenous.
- NumPy array are faster and more compact than Python lists.
- An array consumes less space and is convenient to use.
- NumPy uses much less memory to store data and it provides a mechanism of specifying the data types. This allows the code to be optimized even further.

## Array

- The elements are of the same type, referred to as the array `dtype`.
- The `rank` of the array is the number of dimensions.
- The `shape` of the array is a tuple of integers giving the size of the array along each dimension.

### 1D array, 2D array, ndarray (N-dimensional array), vector, matrix

- 1D array: Single axis, shape `(N,)`.
- 2D array: Two axes, shape `(M, N)`.
- 3D array: Three axes, shape `(P, M, N)`.

- The NumPy `ndarray` class is used to represent both matrices and vectors.
- A `vector` is an array with a single dimension (there's no difference between row and columns) while a `matrix` refers to an array with 2-dimensions.
- For `3-D` or higher dimensional arrays, the term `tensor` is also commonly used.
- In NumPy, dimensions are called `axes`.

#### How to create a basic array

- `np.array()`, `np.zeros()`, `np.ones()`, `np.empty()`, `np.arange()`, `np.linspace()`, `dtype`

- To create a NumPy array, you can use the function `np.array()`.
- The function `empty` creates an array whose initial content is random and depends on the state of the memory. The reason to use `empty` over `zeros` (or something similar) is speed - just make sure to fill every element afterwards!

### Adding, removing, and sorting elements

- This section covers `np.sort()`, `np.concatenate()`

- Sorting an element is simple with `np.sort()`. You can specify the axis, kind, and order when you call the function.
- In addition to sort, which returns a sorted copy of an array, you can use:

    1. `argsort`, which is an indirect sort along a specified axis,
    2. `lexsort`, which is an indirect stable sort on multiple keys,
    3. `searchsorted`, which will find elements in a sorted array, and
    4. `partition`, which is a partial sort.

- You can concatenate 2 or more with `np.concatenate()`.

### Sizes and Shape of Array

- `ndarray.size` will tell you the total number of elements of the array. This is the product of the elements of the array’s shape.
ndarray.ndim will tell you the number of axes, or dimensions, of the array.

- `ndarray.size` will tell you the total number of elements of the array. This is the product of the elements of the array’s shape.

- `ndarray.shape` will display a tuple of integers that indicate the number of elements stored along each dimension of the array. If, for example, you have a 2-D array with 2 rows and 3 columns, the shape of your array is (2, 3).

### Reshaping an array

- Using `arr.reshape()` will give a new shape to an array without changing the data. Just remember that when you use the reshape method, the array you want to produce needs to have the same number of elements as the original array. If you start with an array with 12 elements, you’ll need to make sure that your new array also has a total of 12 elements.

### Converting a 1-D array into a 2-D array

- You can use np.newaxis and np.expand_dims to increase the dimensions of your existing array.

- Using np.newaxis will increase the dimensions of your array by one dimension when used once. This means that a 1D array will become a 2D array, a 2D array will become a 3D array, and so on.
