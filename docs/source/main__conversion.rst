Reorganizing Data
=================
CalSciPy contains several simple utility functions for converting data between several common forms.

Converting Matrices and Tensors
```````````````````````````````
Often times one might want convert a matrix into a tensor. For example, perhaps one wants to convert a neurons x sample
matrix consisting an entire session into a trials x neurons x trial samples tensor.

.. code-block:: python

   from CalSciPy.conversion import matrix_to_tensor, tensor_to_matrix

   data = np.ones((5, 3600))
   samples_per_trial = 600
   tensor_form = matrix_to_tensor(data, samples_per_trial)

   >>>print(f"{tensor_form.shape=})
   tensor_form.shape=(6, 5, 600)

   matrix_form = tensor_to_matrix(tensor_form)

   >>>print(f"{matrix_form.shape=})
   matrix_form.shape=(5, 3600)
