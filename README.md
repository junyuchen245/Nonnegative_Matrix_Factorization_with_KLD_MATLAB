# Nonnegative_Matrix_Factorization_with_KLD_MATLAB

This program is a MATLAB implementation of the NMF on the <a href="https://github.com/marinkaz/nimfa/tree/master/nimfa/datasets/ORL_faces"> face data</a>.

The function 'nmf' does the original NMF algorithm with the KL-Divergence (shown below) minimization, and the function 'ssnmf' does the sparsity imposed NMF.

KL-Divergence:
![](https://github.com/junyuchen245/Nonnegative_Matrix_Factorization_with_KLD_MATLAB/blob/master/NMF/KLD.png)

Matrix update is given by:
![](https://github.com/junyuchen245/Nonnegative_Matrix_Factorization_with_KLD_MATLAB/blob/master/NMF/matrix_update.png)


An example result:
![](https://github.com/junyuchen245/Nonnegative_Matrix_Factorization_with_KLD_MATLAB/blob/master/NMF/example_img.jpg)
