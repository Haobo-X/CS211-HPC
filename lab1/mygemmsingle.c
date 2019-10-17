#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "util.h"


void dgemm2_nif(const double* A, const double* B, double* C, const int n)
{
    int i = 0;
    for (i = 0; i < n; i += 2)
    {
        int j = 0;
        for (j = 0; j < n; j += 2)
        {
            register double C_0_0 = C[i * n + j];
            register double C_1_0 = C[(i + 1) * n + j];
            register double C_0_1 = C[i * n + (j + 1)];
            register double C_1_1 = C[(i + 1) * n + (j + 1)];

            int k = 0;
            for (k = 0; k < n; k += 2)
            {
                register double A_0_0 = A[i * n + k];
                register double A_1_0 = A[(i + 1) * n + k];
                register double A_0_1 = A[i * n + (k + 1)];
                register double A_1_1 = A[(i + 1) * n + (k + 1)];

                register double B_0_0 = B[k * n + j];
                register double B_1_0 = B[(k + 1) * n + j];
                register double B_0_1 = B[k * n + (j + 1)];
                register double B_1_1 = B[(k + 1) * n + (j + 1)];

                C_0_0 += A_0_0 * B_0_0 + A_0_1 * B_1_0;
                C_1_0 += A_1_0 * B_0_0 + A_1_1 * B_1_0;
                C_0_1 += A_0_0 * B_0_1 + A_0_1 * B_1_1;
                C_1_1 += A_1_0 * B_0_1 + A_1_1 * B_1_1;
            }

            C[i * n + j] = C_0_0;
            C[(i + 1) * n + j] = C_1_0;
            C[i * n + (j + 1)] = C_0_1;
            C[(i + 1) * n + (j + 1)] = C_1_1;
        }
    }
}

void dgemm2_2x2_v2_nif(const double* A, const double* B, double* C, const int n)
{
    int i = 0;
    for (i = 0; i < n; i += 2)
    {
        int j = 0;
        for (j = 0; j < n; j += 2)
        {
            register double C_0_0 = C[i * n + j];
            register double C_1_0 = C[(i + 1) * n + j];

            register double C_0_1 = C[i * n + (j + 1)];
            register double C_1_1 = C[(i + 1) * n + (j + 1)];

            int k = 0;
            for (k = 0; k < n; k++)
            {
                register double A_0_M = A[i * n + k];
                register double A_1_M = A[(i + 1) * n + k];

                register double B_M = B[k * n + j];
                C_0_0 += A_0_M * B_M;
                C_1_0 += A_1_M * B_M;

                B_M = B[k * n + (j + 1)];
                C_0_1 += A_0_M * B_M;
                C_1_1 += A_1_M * B_M;
            }

            C[i * n + j] = C_0_0;
            C[(i + 1) * n + j] = C_1_0;

            C[i * n + (j + 1)] = C_0_1;
            C[(i + 1) * n + (j + 1)] = C_1_1;
        }
    }
}

void dgemm3_nif(const double* A, const double* B, double* C, const int n)
{
    //block 3X4, 12 for C, 3 for A, 1 for B, total 16;
    int i = 0;
    for (i = 0; i < n; i += 3)
    {
        int j = 0;
        for (j = 0; j < n; j += 4)
        {
            register double C_0_0 = C[i * n + j];
            register double C_1_0 = C[(i + 1) * n + j];
            register double C_2_0 = C[(i + 2) * n + j];

            register double C_0_1 = C[i * n + (j + 1)];
            register double C_1_1 = C[(i + 1) * n + (j + 1)];
            register double C_2_1 = C[(i + 2) * n + (j + 1)];

            register double C_0_2 = C[i * n + (j + 2)];
            register double C_1_2 = C[(i + 1) * n + (j + 2)];
            register double C_2_2 = C[(i + 2) * n + (j + 2)];

            register double C_0_3 = C[i * n + (j + 3)];
            register double C_1_3 = C[(i + 1) * n + (j + 3)];
            register double C_2_3 = C[(i + 2) * n + (j + 3)];

            int k = 0;
            for (k = 0; k < n; k++)
            {
                register double A_0_M = A[i * n + k];
                register double A_1_M = A[(i + 1) * n + k];
                register double A_2_M = A[(i + 2) * n + k];

                register double B_M = B[k * n + j];
                C_0_0 += A_0_M * B_M;
                C_1_0 += A_1_M * B_M;
                C_2_0 += A_2_M * B_M;

                B_M = B[k * n + (j + 1)];
                C_0_1 += A_0_M * B_M;
                C_1_1 += A_1_M * B_M;
                C_2_1 += A_2_M * B_M;

                B_M = B[k * n + (j + 2)];
                C_0_2 += A_0_M * B_M;
                C_1_2 += A_1_M * B_M;
                C_2_2 += A_2_M * B_M;

                B_M = B[k * n + (j + 3)];
                C_0_3 += A_0_M * B_M;
                C_1_3 += A_1_M * B_M;
                C_2_3 += A_2_M * B_M;
            }

            C[i * n + j] = C_0_0;
            C[(i + 1) * n + j] = C_1_0;
            C[(i + 2) * n + j] = C_2_0;

            C[i * n + (j + 1)] = C_0_1;
            C[(i + 1) * n + (j + 1)] = C_1_1;
            C[(i + 2) * n + (j + 1)] = C_2_1;

            C[i * n + (j + 2)] = C_0_2;
            C[(i + 1) * n + (j + 2)] = C_1_2;
            C[(i + 2) * n + (j + 2)] = C_2_2;

            C[i * n + (j + 3)] = C_0_3;
            C[(i + 1) * n + (j + 3)] = C_1_3;
            C[(i + 2) * n + (j + 3)] = C_2_3;
        }
    }
}

void dgemm3_3x3_nif(const double* A, const double* B, double* C, const int n)
{
    int i = 0;
    for (i = 0; i < n; i += 3)
    {
        int j = 0;
        for (j = 0; j < n; j += 3)
        {
            register double C_0_0 = C[i * n + j];
            register double C_1_0 = C[(i + 1) * n + j];
            register double C_2_0 = C[(i + 2) * n + j];

            register double C_0_1 = C[i * n + (j + 1)];
            register double C_1_1 = C[(i + 1) * n + (j + 1)];
            register double C_2_1 = C[(i + 2) * n + (j + 1)];

            register double C_0_2 = C[i * n + (j + 2)];
            register double C_1_2 = C[(i + 1) * n + (j + 2)];
            register double C_2_2 = C[(i + 2) * n + (j + 2)];

            int k = 0;
            for (k = 0; k < n; k++)
            {
                register double A_0_M = A[i * n + k];
                register double A_1_M = A[(i + 1) * n + k];
                register double A_2_M = A[(i + 2) * n + k];

                register double B_M_0 = B[k * n + j];
                register double B_M_1 = B[k * n + (j + 1)];
                register double B_M_2 = B[k * n + (j + 2)];

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
            C[(i + 1) * n + j] = C_1_0;
            C[(i + 2) * n + j] = C_2_0;

            C[i * n + (j + 1)] = C_0_1;
            C[(i + 1) * n + (j + 1)] = C_1_1;
            C[(i + 2) * n + (j + 1)] = C_2_1;

            C[i * n + (j + 2)] = C_0_2;
            C[(i + 1) * n + (j + 2)] = C_1_2;
            C[(i + 2) * n + (j + 2)] = C_2_2;
        }
    }
}

void dgemm3_3x3_v2_nif(const double* A, const double* B, double* C, const int n)
{
    int i = 0;
    for (i = 0; i < n; i += 3)
    {
        int j = 0;
        for (j = 0; j < n; j += 3)
        {
            register double C_0_0 = C[i * n + j];
            register double C_1_0 = C[(i + 1) * n + j];
            register double C_2_0 = C[(i + 2) * n + j];

            register double C_0_1 = C[i * n + (j + 1)];
            register double C_1_1 = C[(i + 1) * n + (j + 1)];
            register double C_2_1 = C[(i + 2) * n + (j + 1)];

            register double C_0_2 = C[i * n + (j + 2)];
            register double C_1_2 = C[(i + 1) * n + (j + 2)];
            register double C_2_2 = C[(i + 2) * n + (j + 2)];

            int k = 0;
            for (k = 0; k < n; k++)
            {
                register double A_0_M = A[i * n + k];
                register double A_1_M = A[(i + 1) * n + k];
                register double A_2_M = A[(i + 2) * n + k];

                register double B_M = B[k * n + j];
                C_0_0 += A_0_M * B_M;
                C_1_0 += A_1_M * B_M;
                C_2_0 += A_2_M * B_M;

                B_M = B[k * n + (j + 1)];
                C_0_1 += A_0_M * B_M;
                C_1_1 += A_1_M * B_M;
                C_2_1 += A_2_M * B_M;

                B_M = B[k * n + (j + 2)];
                C_0_2 += A_0_M * B_M;
                C_1_2 += A_1_M * B_M;
                C_2_2 += A_2_M * B_M;
            }

            C[i * n + j] = C_0_0;
            C[(i + 1) * n + j] = C_1_0;
            C[(i + 2) * n + j] = C_2_0;

            C[i * n + (j + 1)] = C_0_1;
            C[(i + 1) * n + (j + 1)] = C_1_1;
            C[(i + 2) * n + (j + 1)] = C_2_1;

            C[i * n + (j + 2)] = C_0_2;
            C[(i + 1) * n + (j + 2)] = C_1_2;
            C[(i + 2) * n + (j + 2)] = C_2_2;
        }
    }
}

void dgemm4_4x4_nif(const double* A, const double* B, double* C, const int n)
{
    int i = 0;
    for (i = 0; i < n; i += 4)
    {
        int j = 0;
        for (j = 0; j < n; j += 4)
        {
            register double C_0_0 = C[i * n + j];
            register double C_1_0 = C[(i + 1) * n + j];
            register double C_2_0 = C[(i + 2) * n + j];
            register double C_3_0 = C[(i + 3) * n + j];

            register double C_0_1 = C[i * n + (j + 1)];
            register double C_1_1 = C[(i + 1) * n + (j + 1)];
            register double C_2_1 = C[(i + 2) * n + (j + 1)];
            register double C_3_1 = C[(i + 3) * n + (j + 1)];

            register double C_0_2 = C[i * n + (j + 2)];
            register double C_1_2 = C[(i + 1) * n + (j + 2)];
            register double C_2_2 = C[(i + 2) * n + (j + 2)];
            register double C_3_2 = C[(i + 3) * n + (j + 2)];

            register double C_0_3 = C[i * n + (j + 3)];
            register double C_1_3 = C[(i + 1) * n + (j + 3)];
            register double C_2_3 = C[(i + 2) * n + (j + 3)];
            register double C_3_3 = C[(i + 3) * n + (j + 3)];

            int k = 0;
            for (k = 0; k < n; k++)
            {
                register double A_0_M = A[i * n + k];
                register double A_1_M = A[(i + 1) * n + k];
                register double A_2_M = A[(i + 2) * n + k];
                register double A_3_M = A[(i + 3) * n + k];


                register double B_M_0 = B[k * n + j];
                register double B_M_1 = B[k * n + (j + 1)];
                register double B_M_2 = B[k * n + (j + 2)];
                register double B_M_3 = B[k * n + (j + 3)];

                C_0_0 += A_0_M * B_M_0;
                C_1_0 += A_1_M * B_M_0;
                C_2_0 += A_2_M * B_M_0;
                C_3_0 += A_3_M * B_M_0;

                C_0_1 += A_0_M * B_M_1;
                C_1_1 += A_1_M * B_M_1;
                C_2_1 += A_2_M * B_M_1;
                C_3_1 += A_3_M * B_M_1;

                C_0_2 += A_0_M * B_M_2;
                C_1_2 += A_1_M * B_M_2;
                C_2_2 += A_2_M * B_M_2;
                C_3_2 += A_3_M * B_M_2;

                C_0_3 += A_0_M * B_M_3;
                C_1_3 += A_1_M * B_M_3;
                C_2_3 += A_2_M * B_M_3;
                C_3_3 += A_3_M * B_M_3;
            }

            C[i * n + j] = C_0_0;
            C[(i + 1) * n + j] = C_1_0;
            C[(i + 2) * n + j] = C_2_0;
            C[(i + 3) * n + j] = C_3_0;

            C[i * n + (j + 1)] = C_0_1;
            C[(i + 1) * n + (j + 1)] = C_1_1;
            C[(i + 2) * n + (j + 1)] = C_2_1;
            C[(i + 3) * n + (j + 1)] = C_3_1;

            C[i * n + (j + 2)] = C_0_2;
            C[(i + 1) * n + (j + 2)] = C_1_2;
            C[(i + 2) * n + (j + 2)] = C_2_2;
            C[(i + 3) * n + (j + 2)] = C_3_2;

            C[i * n + (j + 3)] = C_0_3;
            C[(i + 1) * n + (j + 3)] = C_1_3;
            C[(i + 2) * n + (j + 3)] = C_2_3;
            C[(i + 3) * n + (j + 3)] = C_3_3;
        }
    }
}
int randomize_matrix(double *A, const int m, const int n)
{
	srand(time(NULL));
	int i, j;
	for (i = 0; i < m; i++)
	{
		for (j = 0; A + i * n + j && j < n; j++)
		{
			A[i * n + j] = (double)(rand() % 100) + 0.01 * (rand() % 100);
			if ( (rand() % 2) == 0 )
			{
				A[i * n + j] *= -1.;
			}
		}
        if (j != n) return -1;
	}
    return 0;
}

int main(int argc, char *argv[])
{
    printf("\n*********** Register Reuse ***********\n");
    int i, j;

    int curr_dim = 4080;

    double t0, t1;

    double *A = (double *)malloc(sizeof(double) * curr_dim * curr_dim);
    double *B = (double *)malloc(sizeof(double) * curr_dim * curr_dim);

    if ( randomize_matrix(A, curr_dim, curr_dim) ) return -1;
    if ( randomize_matrix(B, curr_dim, curr_dim) ) return -1;

    double dur = t1 - t0;
    C[i] = (double *)malloc(sizeof(double) * curr_dim * curr_dim);
    t0 = get_sec();

    dgemm2_nif(A,B,C,curr_dim);

    t1 = get_sec();
    dur = t1 - t0;

    printf("dgemm2_nif time is %8.5f second(s).\n", dur);

        t0 = get_sec();

    dgemm2_2x2_v2_nif(A,B,C,curr_dim);

    t1 = get_sec();
    dur = t1 - t0;

    printf("dgemm2_2x2_v2_nif time is %8.5f second(s).\n", dur);

        t0 = get_sec();

    dgemm3_nif(A,B,C,curr_dim);

    t1 = get_sec();
    dur = t1 - t0;

    printf("dgemm3_nif time is %8.5f second(s).\n", dur);

        t0 = get_sec();

    dgemm3_3x3_nif(A,B,C,curr_dim);

    t1 = get_sec();
    dur = t1 - t0;

    printf("dgemm3_3x3_nif time is %8.5f second(s).\n", dur);

    t0 = get_sec();

    dgemm3_3x3_v2_nif(A,B,C,curr_dim);

    t1 = get_sec();
    dur = t1 - t0;

    printf("dgemm3_3x3_v2_nif time is %8.5f second(s).\n", dur);

    t0 = get_sec();

    dgemm4_4x4_nif(A,B,C,curr_dim);

    t1 = get_sec();
    dur = t1 - t0;

    printf("dgemm4_4x4_nif time is %8.5f second(s).\n", dur);


    free(A);
    free(B);
    return 0;
}