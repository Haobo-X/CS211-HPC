#include "../include/for_you_to_do.h"
/**
 * 
 * this function computes LU factorization
 * for a square matrix
 * 
 * syntax 
 *  
 *  input : 
 *      A     n by n , square matrix
 *      ipiv  1 by n , vector
 *      n            , length of vector / size of matrix
 *  
 *  output :
 *      return -1 : if the matrix A is singular (max pivot == 0)
 *      return  0 : return normally 
 * 
 **/
int mydgetrf(double *A, int *ipiv, int n) 
{
    /* add your code here */

    return 0;
}

/**
 * 
 * this function computes triangular matrix - vector solver
 * for a square matrix . according to lecture slides, this
 * function computes forward AND backward subtitution in the
 * same function.
 * 
 * syntax 
 *  
 *  input :
 *      UPLO  'L' or 'U' , denotes whether input matrix is upper
 *                         lower triangular . ( forward / backward
 *                         substitution )
 * 
 *      A     n by n     , square matrix
 * 
 *      B     1 by n     , vector
 * 
 *      ipiv  1 by n     , vector , denotes interchanged index due
 *                                  to pivoting by mydgetrf()
 * 
 *      n                , length of vector / size of matrix
 *  
 *  output :
 *      none
 * 
 **/
void mydtrsv(char UPLO, double *A, double *B, int n, int *ipiv)
{
    /* add your code here */
    return;
}

/**
 * 
 * Same function as what you used in lab1, cache_part4.c : optimal( ... ).
 * 
 **/
void mydgemm(double *A, double *B, double *C, int n, int i, int j, int k, int b)
{
    /* add your code here */
    /* please just copy from your lab1 function optimal( ... ) */
    int i = 0;
    for (i = 0; i < n; i += b)
    {
        int j = 0;
        for (j = 0; j < n; j += b)
        {
            int k = 0;
            for (k = 0; k < n; k += b)
            {
                int i1 = 0;
                for (i1 = i; i1 < (i + b > n? n : (i + b)); i1 += 3)
                {
                    int j1 = 0;
                    for (j1 = j; j1 < (j + b > n? n : (j + b)); j1 += 3)
                    {
                        register double C_0_0 = C[i1 * n + j1];
                        register double C_1_0 = C[(i1 + 1) * n + j1];
                        register double C_2_0 = C[(i1 + 2) * n + j1];

                        register double C_0_1 = C[i1 * n + (j1 + 1)];
                        register double C_1_1 = C[(i1 + 1) * n + (j1 + 1)];
                        register double C_2_1 = C[(i1 + 2) * n + (j1 + 1)];

                        register double C_0_2 = C[i1 * n + (j1 + 2)];
                        register double C_1_2 = C[(i1 + 1) * n + (j1 + 2)];
                        register double C_2_2 = C[(i1 + 2) * n + (j1 + 2)];

                        int k1 = 0;
                        for (k1 = k; k1 < (k + b > n? n : (k + b)); k1++)
                        {
                            register double A_0_M = A[i1 * n + k1];
                            register double A_1_M = A[(i1 + 1) * n + k1];
                            register double A_2_M = A[(i1 + 2) * n + k1];

                            register double B_M =  B[k1 * n + j1];
                            C_0_0 += A_0_M * B_M;
                            C_1_0 += A_1_M * B_M;
                            C_2_0 += A_2_M * B_M;

                            B_M = B[k1 * n + (j1 + 1)];
                            C_0_1 += A_0_M * B_M;
                            C_1_1 += A_1_M * B_M;
                            C_2_1 += A_2_M * B_M;

                            B_M = B[k1 * n + (j1 + 2)];
                            C_0_2 += A_0_M * B_M;
                            C_1_2 += A_1_M * B_M;
                            C_2_2 += A_2_M * B_M;

                        }
                        C[i1 * n + j1] = C_0_0;
                        C[(i1 + 1) * n + j1] = C_1_0;
                        C[(i1 + 2) * n + j1] = C_2_0;

                        C[i1 * n + (j1 + 1)] = C_0_1;
                        C[(i1 + 1) * n + (j1 + 1)] = C_1_1;
                        C[(i1 + 2) * n + (j1 + 1)] = C_2_1;

                        C[i1 * n + (j1 + 2)] = C_0_2;
                        C[(i1 + 1) * n + (j1 + 2)] = C_1_2;
                        C[(i1 + 2) * n + (j1 + 2)] = C_2_2;
                    
                    }
                }
            }
        }
    }
    return;
}

/**
 * 
 * this function computes triangular matrix - vector solver
 * for a square matrix using block gepp introduced in course
 * lecture .
 * 
 * just implement the block algorithm you learned in class.
 * 
 * syntax 
 *  
 *  input :
 *      UPLO  'L' or 'U' , denotes whether input matrix is upper
 *                         lower triangular . ( forward / backward
 *                         substitution )
 * 
 *      A     n by n     , square matrix
 * 
 *      B     1 by n     , vector
 * 
 *      ipiv  1 by n     , vector , denotes interchanged index due
 *                                  to pivoting by mydgetrf()
 * 
 *      n                , length of vector / size of matrix
 *  
 *  output :
 *      return -1 : if the matrix A is singular (max pivot == 0)
 *      return  0 : return normally 
 * 
 **/
int mydgetrf_block(double *A, int *ipiv, int n, int b) 
{
    return 0;
}

