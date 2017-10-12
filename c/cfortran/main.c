#include <stdio.h>
#include <stdlib.h>

extern void ssaupd_(int *ido, char *bmat, int *n, char *which,
                   int *nev, double *tol, double *resid,
                   int *ncv, double *V, int *ldv,
                   int *iparam, int *ipntr, double *workd,
                   double *workl, int *lworkl, int *info);

int main(){

    double bmat[] = { 1.0, 0.8, 0.7,
                     0.8, 1.0, 0.6,
                     0.7, 0.6, 1.0 };
    int maxiter = 30;

    // Arguments for Fortran function
    int ido = 0;                // Integer.         Reverse communication flag.
    char bmat = 'I';            // Character*1.     BMAT specifies the type of the matrix B that defines the semi-inner product for the operator OP.
    int numcols = 3;            // Integer          Dimension of the eigenproblem.
    char which[2] = {'L', 'M'}; // Character*2.     Specify which of the Ritz values of OP to compute.
    int numeigs = 3;            // Integer.         Number of eigenvalues of OP to be computed. 0 < NEV < N
    double tol = 1e-13;         // Real scalar      Stopping criterion
    double resid[3] = {0};      // Real array       Final residual vector
    int ncv = 3;                // Integer          Number of columns of the matrix V (less than or equal to N)        
    double* v = (double*) malloc(numcols*ncv * sizeof(double)); // Real array       This will indicate how many Lanczos vectors are generated at each iteration. 
    int iparam[11] = {1, 0, maxiter, 1, 0, 0, 1, 0, 0, 0, 0}; // Integer array       The shifts selected at each iteration are used to restart the Arnoldi iteration in an implicit fashion.
    int ipntr[11];             // Integer array    Pointer to mark the starting locations in the WORKD and WORKL arrays for matrices/vectors used by the Lanczos iteration.
    double* workd = (double*) malloc(3*numcols*sizeof(double)); // Real array      Distributed array to be used in the basic Arnoldi iteration  for reverse communication. 
    int lworkl = ncv*(ncv + 8); // Integer        Must be at least NCV**2 + 8*NCV 
    double *workl = (double*) malloc(lworkl*sizeof(double)); // Real array     Private (replicated) array on each PE or array allocated on the front end.  See Data Distribution Note below.
    int arpack_info = 0;        // Integer         

    printf("Before function call!\n");
    ssaupd_(&ido, &bmat, &numcols, which,
            &numeigs, &tol, resid, 
            &ncv, v, &numcols,
            iparam, ipntr, workd, 
            workl, &lworkl, &arpack_info);
    printf("After function call!\n");
}