#include "../include/for_you_to_do.h"
#include <math.h>
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
void swap(double* A, double* tmpr, int n, int r1, int r2)
{
    memcpy(tmpr, A + r1 * n, n * sizeof(double));
    memcpy(A + r1 * n, A + r2 * n, n * sizeof(double));
    memcpy(A + r2 * n, tmpr, n * sizeof(double));
}

int mydgetrf(double* A, int* ipiv, int n)
{
    /* add your code here */
    int i, j, k;
    double* tmpr = (double*)malloc(sizeof(double) * n);
    for (i = 0; i < n; i++)
    {
        int maxidx = i;
        double max = fabs(A[i * n + i]);
        for (j = i + 1; j < n; j++)
        {
            double tmp = fabs(A[j * n + i]);
            if (tmp - max > 1e-6)
            {
                maxidx = j;
                max = tmp;
            }
        }

        //too small pivot is also unacceptable
        if (fabs(max - 0.0) < 1e-3) 
            return -1;

        if (maxidx != i)
        {
            ipiv[maxidx] = ipiv[maxidx] ^ ipiv[i];
            ipiv[i] = ipiv[maxidx] ^ ipiv[i];
            ipiv[maxidx] = ipiv[maxidx] ^ ipiv[i];

            swap(A, tmpr, n, i, maxidx);
        }

        //do factorization
        for (j = i + 1; j < n; j++)
        {
            A[j * n + i] = A[j * n + i] / A[i * n + i];
            double A_j = A[j * n + i];
            for (k = i + 1; k < n; k++)
            {
                A[j * n + k] -= A_j * A[i * n + k];
            }
        }
    }
    free(tmpr);
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
void mydtrsv(char UPLO, double* A, double* B, int n, int* ipiv)
{
    /* add your code here */
    double *newB = (double*) malloc(n * sizeof(double));
    int i, j;
    if (UPLO == 'L')
    {
        for (i = 0; i < n; i++)
        {
            newB[i] = B[ipiv[i]];
        }

        for (i = 0; i < n; i++)
        {
            double sub = newB[i];
            for (j = 0; j < i; j++)
            {
                sub -= B[j] * A[i * n + j];
            }
            B[i] = sub;
        }
    }
    else
    {//avoid cache miss
        for (i = n-1; i >=0; i--)
        {
            double sum = 0;
            for (j = i+1; j < n; j++)
            {
                sum += B[j] * A[i * n + j];
            }
            B[i] = (B[i] - sum) / A[i * n + i];
        }
    }
    free(newB);
    return;
}


/**
 * 
 * Same function as what you used in lab1, cache_part4.c : optimal( ... ).
 * 
 **/
void mydgemm(const double* A, const double* B, double* C, const int m, const int p, const int n, const int b)
{//mxp pxn = mxn
    int i = 0;
    for (i = 0; i < m; i += b)
    {
        int j = 0;
        for (j = 0; j < n; j += b)
        {
            int k = 0;
            for (k = 0; k < p; k += b)
            {
                int i1 = 0;
                for (i1 = i; i1 < (i + b > m? m : (i + b)); i1 += 3)
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
                        for (k1 = k; k1 < (k + b > p? p : (k + b)); k1++)
                        {
                            register double A_0_M = A[i1 * p + k1];
                            register double A_1_M = A[(i1 + 1) * p + k1];
                            register double A_2_M = A[(i1 + 2) * p + k1];

                            register double B_M =  B[k1 * p + j1];
                            C_0_0 += A_0_M * B_M;
                            C_1_0 += A_1_M * B_M;
                            C_2_0 += A_2_M * B_M;

                            B_M = B[k1 * p + (j1 + 1)];
                            C_0_1 += A_0_M * B_M;
                            C_1_1 += A_1_M * B_M;
                            C_2_1 += A_2_M * B_M;

                            B_M = B[k1 * p + (j1 + 2)];
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
}

void mydgemm_sub(double *A, double *B, double *C, int n, int i, int j, int k, int b)
{
    /* add your code here */
    /* please just copy from your lab1 function optimal( ... ) */
    /*int i = 0;
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
    }*/
    return;
}


/**
 *  
 * this function transposes a square matrix
 * 
 * syntax 
 *  
  *  input : 
 *      A     n by n , a square matrix
 *      n            , length of the whole vector / size of the whole matrix
 * 
 **/

void transpose(double* A, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < i; j++)
        {
            A[i * n + j] = A[i * n + j] ^ A[j * n + i];
            A[j * n + i] = A[i * n + j] ^ A[j * n + i];
            A[i * n + j] = A[i * n + j] ^ A[j * n + i];
        }
    }
}

/**
 *  
 * this function computes LU factorization
 * for a non-square matrix
 * 
 * syntax 
 *  
  *  input : 
 *      A     n by n , a non-square part of the whole matrix
 *      pos          , start position of the whole matrix, both rows and columns.
 *      ipiv  1 by n , vector
 *      n            , length of the whole vector / size of the whole matrix
 *      bm           , number of rows
 *      nm           , number of columns
 *  
 *  output :
 *      return -1 : if the matrix A is singular (max pivot == 0)
 *      return  0 : return normally 
 * 
 **/
int mydgetrf_non_squrare(double* A, int pos, int* ipiv, int n, int bm, int bn)
{
    /* add your code here */
    int i, j, k;
    int bn2 = bm - bn;
    double* tmpr = (double*)malloc(sizeof(double) * n);
    double* LLT = (double*)malloc(sizeof(double) * bn * bn);
    double* AUR = (double*)malloc(sizeof(double) * bn * bn2);
    double* AURD = (double*)malloc(sizeof(double) * bn * bn2);
    double* LL = (double*)malloc(sizeof(double) * bn * bn);
    int* ipivl = (int*)malloc(sizeof(int) * bn);

    for (i = 0; i < bm; i++)
    {
        int maxidx = i;
        double max = fabs(A[i * n + i]);
        for (j = i + 1; j < bm; j++)
        {
            double tmp = fabs(A[j * n + i]);
            if (tmp - max > 1e-6)
            {
                maxidx = j;
                max = tmp;
            }
        }

        //too small pivot is also unacceptable
        if (fabs(max - 0.0) < 1e-3)
            return -1;

        if (maxidx != i)
        {
            int newMaxidx = pos + maxidx;
            int newI      = pos + i;
            ipiv[newMaxidx] = ipiv[newMaxidx] ^ ipiv[newI];
            ipiv[newI] = ipiv[newMaxidx] ^ ipiv[newI];
            ipiv[newMaxidx] = ipiv[newMaxidx] ^ ipiv[newI];

            swap(A-pos, tmpr, n, i, maxidx);
        }

        //do factorization
        for (j = i + 1; j < bm; j++)
        {
            A[j * n + i] = A[j * n + i] / A[i * n + i];
            double A_j = A[j * n + i];
            for (k = i + 1; k < bn; k++)
            {
                A[j * n + k] -= A_j * A[i * n + k];
            }
        }
    }

    memset(LLT, 0, bn * bn * sizeof(double));
    memset(LL, 0, bn * bn * sizeof(double));
    memset(ipivl, 0, bn * sizeof(int));
    memset(AUR, 0, bn * (bn2) * sizeof(double));
    memset(AURD, 0, bn * (bn2) * sizeof(double));

    for (i = 0; i < bn; i++)
    {
        LLT[i * bn + i] = 1;
    }

    for (i = 0; i < bn; i++)
    {
        ipivl[i] = i;
    }

    for (i = 0; i < bn; i++)
    {
        LL[i * bn + i] = 1;
        for (j = 0; j < i; j++)
        {
            LL[i * bn + j] = A[i * n + j];
        }
    }

    for (i = 0; i < bn; i++)
    {
        memcpy(AUR + i * bn2, A + i * n + bn, bn2 * sizeof(double));
    }
    
    for (i = 0; i < bn; i++)
    {
        mydtrsv('L', LL, LLT + i * bn, bn, ipivl);
    }

    transpose(LLT, bn);
    mydgemm(LLT, AUR, AURD, bn, bn, bn2, 126);

    for (i = 0; i < bn; i++)
    {
        memcpy(A + i * n + bn, AURD + i * bn2, bn2 * sizeof(double));
    }

    free(LLT);
    free(LL);
    free(ipivl);
    free(AUR);
    free(AURD);
    free(tmpr);
    return 0;
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
    int i, j, k;

    double* Aptr = A;
    for (i = 0; i < n; i += b)
    {
        Aptr += b * n + b;
        mydgetrf(Aptr, i, iptv, n, n - i, b);
    }
    return 0;
}

