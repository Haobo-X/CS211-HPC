#include "mygemm.h"

/**
 * 
 * Implement all functions here in this file.
 * Do NOT change input parameters and return type.
 * 
 **/


void dgemm0(const double* A, const double* B, double* C, const int n)
{
    int i = 0;
    for (i = 0; i < n; i++)
    {
        int j = 0;
        for (j = 0; j < n; j++)
        {
            int k = 0;
            for (k = 0; k < n; k++)
            {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void dgemm1(const double* A, const double* B, double* C, const int n)
{
    int i = 0;
    for (i = 0; i < n; i++)
    {
        int j = 0;
        for (j = 0; j < n; j++)
        {
            register double C_i_j = C[i * n + j];
            int k = 0;
            for (k = 0; k < n; k++)
            {
                C_i_j += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = C_i_j;
        }
    }
}

void dgemm2(const double* A, const double* B, double* C, const int n)
{
    int i = 0;
    for (i = 0; i < n; i += 2)
    {
        int j = 0;
        for (j = 0; j < n; j += 2)
        {
            register double C_0_0 = C[i * n + j];
            register double C_1_0 = i < (n - 1) ? C[(i + 1) * n + j] : 0;
            register double C_0_1 = j < (n - 1) ? C[i * n + (j + 1)] : 0;
            register double C_1_1 = (i < (n - 1)) && (j < (n - 1))? C[(i + 1) * n + (j + 1)] : 0;

            int k = 0;
            for (k = 0; k < n; k += 2)
            {
                register double A_0_0 = A[i * n + k];
                register double A_1_0 = i < (n - 1) ? A[(i + 1) * n + k] : 0;
                register double A_0_1 = k < (n - 1) ? A[i * n + (k + 1)] : 0;
                register double A_1_1 = (i < (n - 1)) && (k < (n - 1)) ? A[(i + 1) * n + (k + 1)] : 0;

                register double B_0_0 = B[k * n + j];
                register double B_1_0 = k < (n - 1) ? B[(k + 1) * n + j] : 0;
                register double B_0_1 = j < (n - 1) ? B[k * n + (j + 1)] : 0;
                register double B_1_1 = (k < (n - 1)) && (j < (n - 1)) ? B[(k + 1) * n + (j + 1)] : 0;

                C_0_0 += A_0_0 * B_0_0 + A_0_1 * B_1_0;
                C_1_0 += A_1_0 * B_0_0 + A_1_1 * B_1_0;
                C_0_1 += A_0_0 * B_0_1 + A_0_1 * B_1_1;
                C_1_1 += A_1_0 * B_0_1 + A_1_1 * B_1_1;
            }

            C[i * n + j] = C_0_0;
            if (i < (n - 1)) C[(i + 1) * n + j] = C_1_0;
            if (j < (n - 1)) C[i * n + (j + 1)] = C_0_1;
            if (i < (n - 1) && j < (n - 1)) C[(i + 1) * n + (j + 1)] = C_1_1;
        }
    }
}

void dgemm2_2x2_v2(const double* A, const double* B, double* C, const int n)
{
    int i = 0;
    for (i = 0; i < n; i += 2)
    {
        int j = 0;
        for (j = 0; j < n; j += 2)
        {
            register double C_0_0 = C[i * n + j];
            register double C_1_0 = i < (n - 1) ? C[(i + 1) * n + j] : 0;

            register double C_0_1 = j < (n - 1) ? C[i * n + (j + 1)] : 0;
            register double C_1_1 = (i < (n - 1)) && (j < (n - 1)) ? C[(i + 1) * n + (j + 1)] : 0;

            int k = 0;
            for (k = 0; k < n; k++)
            {
                register double A_0_M = A[i * n + k];
                register double A_1_M = i < (n - 1) ? A[(i + 1) * n + k] : 0;

                register double B_M = B[k * n + j];
                C_0_0 += A_0_M * B_M;
                C_1_0 += A_1_M * B_M;

                B_M = j < (n - 1) ? B[k * n + (j + 1)] : 0;
                C_0_1 += A_0_M * B_M;
                C_1_1 += A_1_M * B_M;
            }

            C[i * n + j] = C_0_0;
            if (i < (n - 1)) C[(i + 1) * n + j] = C_1_0;

            if (j < (n - 1)) C[i * n + (j + 1)] = C_0_1;
            if (i < (n - 1) && j < (n - 1)) C[(i + 1) * n + (j + 1)] = C_1_1;
        }
    }
}

void dgemm3_3x4(const double* A, const double* B, double* C, const int n)
{
    //block 3X4, 12 for C, 3 for A, 1 for B, total 16;
    int i = 0;
    for (i = 0; i < n; i += 3)
    {
        int j = 0;
        for (j = 0; j < n; j += 4)
        {
            register double C_0_0 = C[i * n + j];
            register double C_1_0 = i < (n - 1) ? C[(i + 1) * n + j] : 0;
            register double C_2_0 = i < (n - 2) ? C[(i + 2) * n + j] : 0;

            register double C_0_1 = j < (n - 1) ? C[i * n + (j + 1)] : 0;
            register double C_1_1 = i < (n - 1) && j < (n - 1) ? C[(i + 1) * n + (j + 1)] : 0;
            register double C_2_1 = i < (n - 2) && j < (n - 1) ? C[(i + 2) * n + (j + 1)] : 0;

            register double C_0_2 = j < (n - 2) ? C[i * n + (j + 2)] : 0;
            register double C_1_2 = i < (n - 1) && j < (n - 2) ? C[(i + 1) * n + (j + 2)] : 0;
            register double C_2_2 = i < (n - 2) && j < (n - 2) ? C[(i + 2) * n + (j + 2)] : 0;

            register double C_0_3 = j < (n - 3) ? C[i * n + (j + 3)] : 0;
            register double C_1_3 = i < (n - 1) && j < (n - 3) ? C[(i + 1) * n + (j + 3)] : 0;
            register double C_2_3 = i < (n - 2) && j < (n - 3) ? C[(i + 2) * n + (j + 3)] : 0;

            int k = 0;
            for (k = 0; k < n; k++)
            {
                register double A_0_M = A[i * n + k];
                register double A_1_M = i < (n - 1) ? A[(i + 1) * n + k] : 0;
                register double A_2_M = i < (n - 2) ? A[(i + 2) * n + k] : 0;

                register double B_M = B[k * n + j];
                C_0_0 += A_0_M * B_M;
                C_1_0 += A_1_M * B_M;
                C_2_0 += A_2_M * B_M;

                B_M = j < (n - 1) ? B[k * n + (j + 1)] : 0;
                C_0_1 += A_0_M * B_M;
                C_1_1 += A_1_M * B_M;
                C_2_1 += A_2_M * B_M;

                B_M = j < (n - 2) ? B[k * n + (j + 2)] : 0;
                C_0_2 += A_0_M * B_M;
                C_1_2 += A_1_M * B_M;
                C_2_2 += A_2_M * B_M;

                B_M = j < (n - 3) ? B[k * n + (j + 3)] : 0;
                C_0_3 += A_0_M * B_M;
                C_1_3 += A_1_M * B_M;
                C_2_3 += A_2_M * B_M;
            }

            C[i * n + j] = C_0_0;
            if (i < (n - 1)) C[(i + 1) * n + j] = C_1_0;
            if (i < (n - 2)) C[(i + 2) * n + j] = C_2_0;

            if (j < (n - 1)) C[i * n + (j + 1)] = C_0_1;
            if (i < (n - 1) && j < (n - 1)) C[(i + 1) * n + (j + 1)] = C_1_1;
            if (i < (n - 2) && j < (n - 1)) C[(i + 2) * n + (j + 1)] = C_2_1;

            if (j < (n - 2)) C[i * n + (j + 2)] = C_0_2;
            if (i < (n - 1) && j < (n - 2)) C[(i + 1) * n + (j + 2)] = C_1_2;
            if (i < (n - 2) && j < (n - 2)) C[(i + 2) * n + (j + 2)] = C_2_2;

            if (j < (n - 3)) C[i * n + (j + 3)] = C_0_3;
            if (i < (n - 1) && j < (n - 3)) C[(i + 1) * n + (j + 3)] = C_1_3;
            if (i < (n - 2) && j < (n - 3)) C[(i + 2) * n + (j + 3)] = C_2_3;
        }
    }
}

void dgemm3(const double* A, const double* B, double* C, const int n)
{
    int i = 0;
    for (i = 0; i < n; i += 3)
    {
        int j = 0;
        for (j = 0; j < n; j += 3)
        {
            register double C_0_0 = C[i * n + j];
            register double C_1_0 = i < (n - 1) ? C[(i + 1) * n + j] : 0;
            register double C_2_0 = i < (n - 2) ? C[(i + 2) * n + j] : 0;

            register double C_0_1 = j < (n - 1) ? C[i * n + (j + 1)] : 0;
            register double C_1_1 = (i < (n - 1)) && (j < (n - 1)) ? C[(i + 1) * n + (j + 1)] : 0;
            register double C_2_1 = (i < (n - 2)) && (j < (n - 1)) ? C[(i + 2) * n + (j + 1)] : 0;

            register double C_0_2 = j < (n - 2) ? C[i * n + (j + 2)] : 0;
            register double C_1_2 = (i < (n - 1)) && (j < (n - 2)) ? C[(i + 1) * n + (j + 2)] : 0;
            register double C_2_2 = (i < (n - 2)) && (j < (n - 2)) ? C[(i + 2) * n + (j + 2)] : 0;

            int k = 0;
            for (k = 0; k < n; k++)
            {
                register double A_0_M = A[i * n + k];
                register double A_1_M = i < (n - 1) ? A[(i + 1) * n + k] : 0;
                register double A_2_M = i < (n - 2) ? A[(i + 2) * n + k] : 0;

                register double B_M_0 = B[k * n + j];
                register double B_M_1 = j < (n - 1) ? B[k * n + (j + 1)] : 0;
                register double B_M_2 = j < (n - 2) ? B[k * n + (j + 2)] : 0;

                C_0_0 += A_0_M * B_M_0;
                C_1_0 += A_1_M * B_M_0;
                C_2_0 += A_2_M * B_M_0;

                C_0_1 += A_0_M * B_M_1;
                C_1_1 += A_1_M * B_M_1;
                C_2_1 += A_2_M * B_M_1;

                C_0_2 += A_0_M * B_M_2;
                C_1_2 += A_1_M * B_M_2;
                C_2_2 += A_2_M * B_M_2;
            }

            C[i * n + j] = C_0_0;
            if (i < (n - 1)) C[(i + 1) * n + j] = C_1_0;
            if (i < (n - 2)) C[(i + 2) * n + j] = C_2_0;

            if (j < (n - 1)) C[i * n + (j + 1)] = C_0_1;
            if (i < (n - 1) && j < (n - 1)) C[(i + 1) * n + (j + 1)] = C_1_1;
            if (i < (n - 2) && j < (n - 1)) C[(i + 2) * n + (j + 1)] = C_2_1;

            if (j < (n - 2)) C[i * n + (j + 2)] = C_0_2;
            if (i < (n - 1) && j < (n - 2)) C[(i + 1) * n + (j + 2)] = C_1_2;
            if (i < (n - 2) && j < (n - 2)) C[(i + 2) * n + (j + 2)] = C_2_2;
        }
    }
}

void dgemm3_3x3_v2(const double* A, const double* B, double* C, const int n)
{
    int i = 0;
    for (i = 0; i < n; i += 3)
    {
        int j = 0;
        for (j = 0; j < n; j += 3)
        {
            register double C_0_0 = C[i * n + j];
            register double C_1_0 = i < (n - 1) ? C[(i + 1) * n + j] : 0;
            register double C_2_0 = i < (n - 2) ? C[(i + 2) * n + j] : 0;

            register double C_0_1 = j < (n - 1) ? C[i * n + (j + 1)] : 0;
            register double C_1_1 = (i < (n - 1)) && (j < (n - 1)) ? C[(i + 1) * n + (j + 1)] : 0;
            register double C_2_1 = (i < (n - 2)) && (j < (n - 1)) ? C[(i + 2) * n + (j + 1)] : 0;

            register double C_0_2 = j < (n - 2) ? C[i * n + (j + 2)] : 0;
            register double C_1_2 = (i < (n - 1)) && (j < (n - 2)) ? C[(i + 1) * n + (j + 2)] : 0;
            register double C_2_2 = (i < (n - 2)) && (j < (n - 2)) ? C[(i + 2) * n + (j + 2)] : 0;

            int k = 0;
            for (k = 0; k < n; k++)
            {
                register double A_0_M = A[i * n + k];
                register double A_1_M = i < (n - 1) ? A[(i + 1) * n + k] : 0;
                register double A_2_M = i < (n - 2) ? A[(i + 2) * n + k] : 0;

                register double B_M = B[k * n + j];
                C_0_0 += A_0_M * B_M;
                C_1_0 += A_1_M * B_M;
                C_2_0 += A_2_M * B_M;

                B_M = j < (n - 1) ? B[k * n + (j + 1)] : 0;
                C_0_1 += A_0_M * B_M;
                C_1_1 += A_1_M * B_M;
                C_2_1 += A_2_M * B_M;

                B_M = j < (n - 2) ? B[k * n + (j + 2)] : 0;
                C_0_2 += A_0_M * B_M;
                C_1_2 += A_1_M * B_M;
                C_2_2 += A_2_M * B_M;
            }

            C[i * n + j] = C_0_0;
            if (i < (n - 1)) C[(i + 1) * n + j] = C_1_0;
            if (i < (n - 2)) C[(i + 2) * n + j] = C_2_0;

            if (j < (n - 1)) C[i * n + (j + 1)] = C_0_1;
            if (i < (n - 1) && j < (n - 1)) C[(i + 1) * n + (j + 1)] = C_1_1;
            if (i < (n - 2) && j < (n - 1)) C[(i + 2) * n + (j + 1)] = C_2_1;

            if (j < (n - 2)) C[i * n + (j + 2)] = C_0_2;
            if (i < (n - 1) && j < (n - 2)) C[(i + 1) * n + (j + 2)] = C_1_2;
            if (i < (n - 2) && j < (n - 2)) C[(i + 2) * n + (j + 2)] = C_2_2;
        }
    }
}

void ijk(const double* A, const double* B, double* C, const int n)
{
    int i = 0;
    for (i = 0; i < n; i++)
    {
        int j = 0;
        for (j = 0; j < n; j++)
        {
            register double C_i_j = C[i * n + j];
            int k = 0;
            for (k = 0; k < n; k++)
            {
                C_i_j += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = C_i_j;
        }
    }
}

void bijk(const double* A, const double* B, double* C, const int n, const int b)
{
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
                for (i1 = i; i1 < i + b && i1 < n; i1++)
                {
                    int j1 = 0;
                    for (j1 = j; j1 < j + b && j1 < n; j1++)
                    {
                        register double C_i1_j1 = C[i1 * n + j1];
                        int k1 = 0;
                        for (k1 = k; k1 < k + b && k1 < n; k1++)
                        {
                            C_i1_j1 += A[i1 * n + k1] * B[k1 * n + j1];
                        }
                        C[i1 * n + j1] = C_i1_j1;
                    }
                }
            }
        }
    }
}

void jik(const double* A, const double* B, double* C, const int n)
{
    int j = 0;
    for (j = 0; j < n; j++)
    {
        int i = 0;
        for (i = 0; i < n; i++)
        {
            register double C_i_j = C[i * n + j];
            int k = 0;
            for (k = 0; k < n; k++)
            {
                C_i_j += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = C_i_j;
        }
    }
}

void bjik(const double* A, const double* B, double* C, const int n, const int b)
{
    int j = 0;
    for (j = 0; j < n; j += b)
    {
        int i = 0;
        for (i = 0; i < n; i += b)
        {
            int k = 0;
            for (k = 0; k < n; k += b)
            {
                int j1 = 0;
                for (j1 = j; j1 < j + b && j1 < n; j1++)
                {
                    int i1 = 0;
                    for (i1 = i; i1 < i + b && i1 < n; i1++)
                    {
                        register double C_i1_j1 = C[i1 * n + j1];
                        int k1 = 0;
                        for (k1 = k; k1 < k + b && k1 < n; k1++)
                        {
                            C_i1_j1 += A[i1 * n + k1] * B[k1 * n + j1];
                        }
                        C[i1 * n + j1] = C_i1_j1;
                    }
                }
            }
        }
    }
}

void kij(const double* A, const double* B, double* C, const int n)
{
    int k = 0;
    for (k = 0; k < n; k++)
    {
        int i = 0;
        for (i = 0; i < n; i++)
        {
            register double A_i_k = A[i * n + k];
            int j = 0;
            for (j = 0; j < n; j++)
            {
                C[i * n + j] += A_i_k * B[k * n + j];
            }
        }
    }
}

void bkij(const double* A, const double* B, double* C, const int n, const int b)
{
    int k = 0;
    for (k = 0; k < n; k += b)
    {
        int i = 0;
        for (i = 0; i < n; i += b)
        {
            int j = 0;
            for (j = 0; j < n; j += b)
            {
                int k1 = 0;
                for (k1 = k; k1 < k + b && k1 < n; k1++)
                {
                    int i1 = 0;
                    for (i1 = i; i1 < i + b && i1 < n; i1++)
                    {
                        register double A_i1_k1 = A[i1 * n + k1];
                        int j1 = 0;
                        for (j1 = j; j1 < j + b && j1 < n; j1++)
                        {
                            C[i1 * n + j1] += A_i1_k1 * B[k1 * n + j1];
                        }
                    }
                }
            }
        }
    }
}


void ikj(const double* A, const double* B, double* C, const int n)
{
    int i = 0;
    for (i = 0; i < n; i++)
    {
        int k = 0;
        for (k = 0; k < n; k++)
        {
            register double A_i_k = A[i * n + k];
            int j = 0;
            for (j = 0; j < n; j++)
            {
                C[i * n + j] += A_i_k * B[k * n + j];
            }
        }
    }
}

void bikj(const double* A, const double* B, double* C, const int n, const int b)
{
    int i = 0;
    for (i = 0; i < n; i += b)
    {
        int k = 0;
        for (k = 0; k < n; k += b)
        {
            int j = 0;
            for (j = 0; j < n; j += b)
            {
                int i1 = 0;
                for (i1 = i; i1 < i + b && i1 < n; i1++)
                {
                    int k1 = 0;
                    for (k1 = k; k1 < k + b && k1 < n; k1++)
                    {
                        register double A_i1_k1 = A[i1 * n + k1];
                        int j1 = 0;
                        for (j1 = j; j1 < j + b && j1 < n; j1++)
                        {
                            C[i1 * n + j1] += A_i1_k1 * B[k1 * n + j1];
                        }
                    }
                }
            }
        }
    }
}

void jki(const double* A, const double* B, double* C, const int n)
{
    int j = 0;
    for (j = 0; j < n; j++)
    {
        int k = 0;
        for (k = 0; k < n; k++)
        {
            register double B_k_j = B[k * n + j];
            int i = 0;
            for (i = 0; i < n; i++)
            {
                C[i * n + j] += A[i * n + k] * B_k_j;
            }
        }
    }
}

void bjki(const double* A, const double* B, double* C, const int n, const int b)
{
    int j = 0;
    for (j = 0; j < n; j += b)
    {
        int k = 0;
        for (k = 0; k < n; k += b)
        {
            int i = 0;
            for (i = 0; i < n; i += b)
            {
                int j1 = 0;
                for (j1 = j; j1 < j + b && j1 < n; j1++)
                {
                    int k1 = 0;
                    for (k1 = k; k1 < k + b && k1 < n; k1++)
                    {
                        register double B_k1_j1 = B[k1 * n + j1];
                        int i1 = 0;
                        for (i1 = i; i1 < i + b && i1 < n; i1++)
                        {
                            C[i1 * n + j1] += A[i1 * n + k1] * B_k1_j1;
                        }
                    }
                }
            }
        }
    }
}

void kji(const double* A, const double* B, double* C, const int n)
{
    int k = 0;
    for (k = 0; k < n; k++)
    {
        int j = 0;
        for (j = 0; j < n; j++)
        {
            register double B_k_j = B[k * n + j];
            int i = 0;
            for (i = 0; i < n; i++)
            {
                C[i * n + j] += A[i * n + k] * B_k_j;
            }
        }
    }
}

void bkji(const double* A, const double* B, double* C, const int n, const int b)
{
    int k = 0;
    for (k = 0; k < n; k += b)
    {
        int j = 0;
        for (j = 0; j < n; j += b)
        {
            int i = 0;
            for (i = 0; i < n; i += b)
            {
                int k1 = 0;
                for (k1 = k; k1 < k + b && k1 < n; k1++)
                {
                    int j1 = 0;
                    for (j1 = j; j1 < j + b && j1 < n; j1++)
                    {
                        register double B_k1_j1 = B[k1 * n + j1];
                        int i1 = 0;
                        for (i1 = i; i1 < i + b && i1 < n; i1++)
                        {
                            C[i1 * n + j1] += A[i1 * n + k1] * B_k1_j1;
                        }
                    }
                }
            }
        }
    }
}


void optimal2(const double* A, const double* B, double* C, const int n, const int b)
{
    int k = 0;
    for (k = 0; k < n; k += b)
    {
        int i = 0;
        for (i = 0; i < n; i += b)
        {
            int j = 0;
            for (j = 0; j < n; j += b)
            {
                int k1 = 0;
                for (k1 = k; k1 < k + b && k1 < n; k1+=3)
                {
                    int i1 = 0;
                    for (i1 = i; i1 < i + b && i1 < n; i1+=3)
                    {
                        register double A_i1_k1 = A[i1 * n + k1];
                        register double A_0_0 = i1 < (i + b) && k1 < (k + b) && i1 < n && k1 < n ? A[i1 * n + k1] : 0;
                        register double A_1_0 = i1 < (i + b - 1) && k1 < (k + b) && i1 < (n - 1) && k1 < n ? A[(i1 + 1) * n + k1] : 0;
                        register double A_2_0 = i1 < (i + b - 2) && k1 < (k + b) && i1 < (n - 2) && k1 < n ? A[(i1 + 2) * n + k1] : 0;

                        register double A_0_1 = i1 < (i + b) && k1 < (k + b - 1) && i1 < n && k1 < (n - 1) ? A[i1 * n + (k1 + 1)] : 0;
                        register double A_1_1 = i1 < (i + b - 1) && k1 < (k + b - 1) && i1 < (n - 1) && k1 < (n - 1) ? A[(i1 + 1) * n + (k1 + 1)] : 0;
                        register double A_2_1 = i1 < (i + b - 2) && k1 < (k + b - 1) && i1 < (n - 2) && k1 < (n - 1) ? A[(i1 + 2) * n + (k1 + 1)] : 0;

                        register double A_0_2 = i1 < (i + b) && k1 < (k + b - 2) && i1 < n && k1 < (n - 2) ? A[i1 * n + (k1 + 2)] : 0;
                        register double A_1_2 = i1 < (i + b - 1) && k1 < (k + b - 2) && i1 < (n - 1) && k1 < (n - 2) ? A[(i1 + 1) * n + (k1 + 2)] : 0;
                        register double A_2_2 = i1 < (i + b - 2) && k1 < (k + b - 2) && i1 < (n - 2) && k1 < (n - 2) ? A[(i1 + 2) * n + (k1 + 2)] : 0;
                        int j1 = 0;
                        for (j1 = j; j1 < j + b && j1 < n; j1++)
                        {
                            register double B_0_M = j1 < (j + b) && k1 < (k + b) && j1 < n && k1 < n ? B[k1 * n + j1] : 0;
                            register double B_1_M = j1 < (j + b) && k1 < (k + b - 1) && j1 < n && k1 < (n - 1) ? B[(k1 + 1) * n + j1] : 0;
                            register double B_2_M = j1 < (j + b) && k1 < (k + b - 2) && j1 < n && k1 < (n - 2) ? B[(k1 + 2) * n + j1] : 0;
                            
                            if (i1 < (i + b) && j1 < (j + b) && i1 < n && j1 < n) C[i1 * n + j1] += A_0_0 * B_0_M + A_0_1 * B_1_M + A_0_2 * B_2_M;
                            if (i1 < (i + b - 1) && j1 < (j + b) && i1 < (n - 1) && j1 < n) C[(i1 + 1) * n + j1] += A_1_0 * B_0_M + A_1_1 * B_1_M + A_1_2 * B_2_M;
                            if (i1 < (i + b - 2) && j1 < (j + b) && i1 < (n - 2) && j1 < n) C[(i1 + 2) * n + j1] += A_2_0 * B_0_M + A_2_1 * B_1_M + A_2_2 * B_2_M;;
                        }
                    }
                }
            }
        }
    }
}

void optimal(const double* A, const double* B, double* C, const int n, const int b)
{
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
				for (i1 = i; i1 < i + b; i1 += 3)
				{
					int j1 = 0;
					for (j1 = j; j1 < j + b; j1 += 3)
					{
						register double C_0_0 = i1 < (i + b) && j1 < (j + b) && i1 < n && j1 < n ? C[i1 * n + j1] : 0;
						register double C_1_0 = i1 < (i + b - 1) && j1 < (j + b) && i1 < (n - 1) && j1 < n ? C[(i1 + 1) * n + j1] : 0;
						register double C_2_0 = i1 < (i + b - 2) && j1 < (j + b) && i1 < (n - 2) && j1 < n ? C[(i1 + 2) * n + j1] : 0;

						register double C_0_1 = i1 < (i + b) && j1 < (j + b - 1) && i1 < n && j1 < (n - 1) ? C[i1 * n + (j1 + 1)] : 0;
						register double C_1_1 = i1 < (i + b - 1) && j1 < (j + b - 1) && i1 < (n - 1) && j1 < (n - 1) ? C[(i1 + 1) * n + (j1 + 1)] : 0;
						register double C_2_1 = i1 < (i + b - 2) && j1 < (j + b - 1) && i1 < (n - 2) && j1 < (n - 1) ? C[(i1 + 2) * n + (j1 + 1)] : 0;

						register double C_0_2 = i1 < (i + b) && j1 < (j + b - 2) && i1 < n && j1 < (n - 2) ? C[i1 * n + (j1 + 2)] : 0;
						register double C_1_2 = i1 < (i + b - 1) && j1 < (j + b - 2) && i1 < (n - 1) && j1 < (n - 2) ? C[(i1 + 1) * n + (j1 + 2)] : 0;
						register double C_2_2 = i1 < (i + b - 2) && j1 < (j + b - 2) && i1 < (n - 2) && j1 < (n - 2) ? C[(i1 + 2) * n + (j1 + 2)] : 0;

						int k1 = 0;
						for (k1 = k; k1 < k + b; k1++)
						{
							register double A_0_M = i1 < (i + b) && k1 < (k + b) && i1 < n && k1 < n ? A[i1 * n + k1] : 0;
							register double A_1_M = i1 < (i + b - 1) && k1 < (k + b) && i1 < (n - 1) && k1 < n ? A[(i1 + 1) * n + k1] : 0;
							register double A_2_M = i1 < (i + b - 2) && k1 < (k + b) && i1 < (n - 2) && k1 < n ? A[(i1 + 2) * n + k1] : 0;

							register double B_M = k1 < (k + b) && j1 < (j + b) && k1 < n && j1 < n ? B[k1 * n + j1] : 0;
							C_0_0 += A_0_M * B_M;
							C_1_0 += A_1_M * B_M;
							C_2_0 += A_2_M * B_M;

							B_M = k1 < (k + b) && j1 < (j + b - 1) && k1 < n && j1 < (n - 1) ? B[k1 * n + (j1 + 1)] : 0;
							C_0_1 += A_0_M * B_M;
							C_1_1 += A_1_M * B_M;
							C_2_1 += A_2_M * B_M;

							B_M = k1 < (k + b) && j1 < (j + b - 2) && k1 < n && j1 < (n - 2) ? B[k1 * n + (j1 + 2)] : 0;
							C_0_2 += A_0_M * B_M;
							C_1_2 += A_1_M * B_M;
							C_2_2 += A_2_M * B_M;

						}

						if (i1 < (i + b) && j1 < (j + b) && i1 < n && j1 < n) C[i1 * n + j1] = C_0_0;
						if (i1 < (i + b - 1) && j1 < (j + b) && i1 < (n - 1) && j1 < n) C[(i1 + 1) * n + j1] = C_1_0;
						if (i1 < (i + b - 2) && j1 < (j + b) && i1 < (n - 2) && j1 < n) C[(i1 + 2) * n + j1] = C_2_0;

						if (i1 < (i + b) && j1 < (j + b - 1) && i1 < n && j1 < (n - 1)) C[i1 * n + (j1 + 1)] = C_0_1;
						if (i1 < (i + b - 1) && j1 < (j + b - 1) && i1 < (n - 1) && j1 < (n - 1)) C[(i1 + 1) * n + (j1 + 1)] = C_1_1;
						if (i1 < (i + b - 2) && j1 < (j + b - 1) && i1 < (n - 2) && j1 < (n - 1)) C[(i1 + 2) * n + (j1 + 1)] = C_2_1;

						if (i1 < (i + b) && j1 < (j + b - 2) && i1 < n && j1 < (n - 2)) C[i1 * n + (j1 + 2)] = C_0_2;
						if (i1 < (i + b - 1) && j1 < (j + b - 2) && i1 < (n - 1) && j1 < (n - 2)) C[(i1 + 1) * n + (j1 + 2)] = C_1_2;
						if (i1 < (i + b - 2) && j1 < (j + b - 2) && i1 < (n - 2) && j1 < (n - 2)) C[(i1 + 2) * n + (j1 + 2)] = C_2_2;
					}
				}
			}
		}
	}
}

void addMatrix(const double* A, const double* B, double* C, const int n)
{
    int i = 0;
    for (i = 0; i < n; i++) 
    {
        int j = 0;
        for (j = 0; j < n; j++) 
        {
            C[i * n + j] = A[i * n + j] + B[i * n + j];
        }
    }
}

void minusMatrix(const double* A, const double* B, double* C, const int n)
{
    int i = 0;
    for (i = 0; i < n; i++) 
    {
        int j = 0;
        for (j = 0; j < n; j++) 
        {
            C[i * n + j] = A[i * n + j] - B[i * n + j];
        }
    }
}

void strassen(const double* A, const double* B, double* C, const int n)
{
    int i, j;
	if ((n & (n - 1)) != 0)
		return;

	if (n == 2)
	{
		C[0] = A[0] * B[0] + A[1] * B[2];
		C[1] = A[0] * B[1] + A[1] * B[3];
		C[2] = A[2] * B[0] + A[3] * B[2];
		C[3] = A[2] * B[1] + A[3] * B[3];
	}
	else
	{
		double* new_A_0_0 = (double*)malloc(sizeof(double) * n / 2 * n / 2);
		double* new_A_0_1 = (double*)malloc(sizeof(double) * n / 2 * n / 2);
		double* new_A_1_0 = (double*)malloc(sizeof(double) * n / 2 * n / 2);
		double* new_A_1_1 = (double*)malloc(sizeof(double) * n / 2 * n / 2);

		for (i = 0; i < n / 2; i++)
		{
			int j = 0;
			for (j = 0; j < n / 2; j++) {
				new_A_0_0[i * n / 2 + j] = A[i * n + j];
				new_A_0_1[i * n / 2 + j] = A[i * n + (j + n / 2)];
				new_A_1_0[i * n / 2 + j] = A[(i + n / 2) * n + j];
				new_A_1_1[i * n / 2 + j] = A[(i + n / 2) * n + (j + n / 2)];
			}
		}

		double* new_B_0_0 = (double*)malloc(sizeof(double) * n / 2 * n / 2);
		double* new_B_0_1 = (double*)malloc(sizeof(double) * n / 2 * n / 2);
		double* new_B_1_0 = (double*)malloc(sizeof(double) * n / 2 * n / 2);
		double* new_B_1_1 = (double*)malloc(sizeof(double) * n / 2 * n / 2);

		for (i = 0; i < n / 2; i++)
		{
			for (j = 0; j < n / 2; j++) {
				new_B_0_0[i * n / 2 + j] = B[i * n + j];
				new_B_0_1[i * n / 2 + j] = B[i * n + (j + n / 2)];
				new_B_1_0[i * n / 2 + j] = B[(i + n / 2) * n + j];
				new_B_1_1[i * n / 2 + j] = B[(i + n / 2) * n + (j + n / 2)];
			}
		}

		double* new_C_0_0 = (double*)malloc(sizeof(double) * n / 2 * n / 2);
		double* new_C_0_1 = (double*)malloc(sizeof(double) * n / 2 * n / 2);
		double* new_C_1_0 = (double*)malloc(sizeof(double) * n / 2 * n / 2);
		double* new_C_1_1 = (double*)malloc(sizeof(double) * n / 2 * n / 2);

		double* M_1 = (double*)malloc(sizeof(double) * n / 2 * n / 2);
		double* M_2 = (double*)malloc(sizeof(double) * n / 2 * n / 2);
		double* M_3 = (double*)malloc(sizeof(double) * n / 2 * n / 2);
		double* M_4 = (double*)malloc(sizeof(double) * n / 2 * n / 2);
		double* M_5 = (double*)malloc(sizeof(double) * n / 2 * n / 2);
		double* M_6 = (double*)malloc(sizeof(double) * n / 2 * n / 2);
		double* M_7 = (double*)malloc(sizeof(double) * n / 2 * n / 2);

		double* LEFT_PLUS = (double*)malloc(sizeof(double) * n / 2 * n / 2);
		double* RIGHT_PLUS = (double*)malloc(sizeof(double) * n / 2 * n / 2);

		addMatrix(new_A_0_0, new_A_1_1, LEFT_PLUS, n / 2);
		addMatrix(new_B_0_0, new_B_1_1, RIGHT_PLUS, n / 2);
		strassen(LEFT_PLUS, RIGHT_PLUS, M_1, n / 2);

		addMatrix(new_A_1_0, new_A_1_1, LEFT_PLUS, n / 2);
		strassen(LEFT_PLUS, new_B_0_0, M_2, n / 2);

		minusMatrix(new_B_0_1, new_B_1_1, RIGHT_PLUS, n / 2);
		strassen(new_A_0_0, RIGHT_PLUS, M_3, n / 2);

		minusMatrix(new_B_1_0, new_B_0_0, RIGHT_PLUS, n / 2);
		strassen(new_A_1_1, RIGHT_PLUS, M_4, n / 2);

		addMatrix(new_A_0_0, new_A_0_1, LEFT_PLUS, n / 2);
		strassen(LEFT_PLUS, new_B_1_1, M_5, n / 2);

		minusMatrix(new_A_1_0, new_A_0_0, LEFT_PLUS, n / 2);
		addMatrix(new_B_0_0, new_B_0_1, RIGHT_PLUS, n / 2);
		strassen(LEFT_PLUS, RIGHT_PLUS, M_6, n / 2);

		minusMatrix(new_A_0_1, new_A_1_1, LEFT_PLUS, n / 2);
		addMatrix(new_B_1_0, new_B_1_1, RIGHT_PLUS, n / 2);
		strassen(LEFT_PLUS, RIGHT_PLUS, M_7, n / 2);

		addMatrix(M_1, M_4, LEFT_PLUS, n / 2);
		addMatrix(LEFT_PLUS, M_7, RIGHT_PLUS, n / 2);
		minusMatrix(RIGHT_PLUS, M_5, new_C_0_0, n / 2);

		addMatrix(M_3, M_5, new_C_0_1, n / 2);

		addMatrix(M_2, M_4, new_C_1_0, n / 2);

		addMatrix(M_1, M_3, LEFT_PLUS, n / 2);
		addMatrix(LEFT_PLUS, M_6, RIGHT_PLUS, n / 2);
		minusMatrix(RIGHT_PLUS, M_2, new_C_1_1, n / 2);

		for (i = 0; i < n / 2; i++)
		{
			for (j = 0; j < n / 2; j++)
			{
				C[i * n + j] = new_C_0_0[i * n / 2 + j];
				C[i * n + (j + n / 2)] = new_C_0_1[i * n / 2 + j];
				C[(i + n / 2) * n + j] = new_C_1_0[i * n / 2 + j];
				C[(i + n / 2) * n + (j + n / 2)] = new_C_1_1[i * n / 2 + j];
			}
		}

		free(new_A_0_0);
		free(new_A_0_1);
		free(new_A_1_0);
		free(new_A_1_1);
		free(new_B_0_0);
		free(new_B_0_1);
		free(new_B_1_0);
		free(new_B_1_1);
		free(new_C_0_0);
		free(new_C_0_1);
		free(new_C_1_0);
		free(new_C_1_1);
		free(M_1);
		free(M_2);
		free(M_3);
		free(M_4);
		free(M_5);
		free(M_6);
		free(M_7);
		free(LEFT_PLUS);
		free(RIGHT_PLUS);
	}
}