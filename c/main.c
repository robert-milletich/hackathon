// Standard library
#include <math.h>
#include <stdio.h>

// GNU scientific library (GSL)
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_matrix.h>

// Macros/constants
#define square(x) (float)((x)*(x))
#define ROWS 1083
#define COLS 64
#define k 2

// Function declarations
double squared_l2_norm(gsl_vector *x, gsl_vector *y, int p);

// Main
int main(){
    
    printf("\nRunning MDS on 6 classes from MNIST data set...\n");

    /******************************/
    /* Initialize data structures */
    /******************************/

    FILE *f, *ff;
    gsl_matrix *data, *P, *J, *scale, *B, *PJ, *E, *X_t, *L;
    gsl_vector *evals;
    gsl_eigen_symmv_workspace *w;


    /******************/
    /* Load test data */
    /******************/

    // Load data matrix from file
    data = gsl_matrix_alloc(ROWS, COLS); 
    f = fopen("mnist_samples.txt", "r"); 
    gsl_matrix_fscanf(f, data);             // Read file into matrix
    fclose(f);
    // gsl_matrix_fprintf(stdout, data, "%.4f");


    /*******************************************************/
    /* Distance matrix based on squared euclidean distance */
    /*******************************************************/

    // Allocate memory
    P = gsl_matrix_alloc(ROWS, ROWS);

    // Outer loop
    for(int i=0; i<ROWS; i++){
        gsl_vector_view v1 = gsl_matrix_row(data, i);        // Grab ith row

        // Inner loop
        for(int j=i+1; j<ROWS; j++){
            gsl_vector_view v2 = gsl_matrix_row(data, j);    // Grab jth row

            // Calculate distance
            double distance = squared_l2_norm(&v1.vector, &v2.vector, COLS);

            // Matrix is symmetric so (i,j) = (j, i)
            gsl_matrix_set(P, i, j, distance);  
            gsl_matrix_set(P, j, i, distance);
        }
    }
    // gsl_matrix_fprintf(stdout, P, "%.4f");
    // printf("Rows P: %zu\n", J->size1);
    // printf("Columns P: %zu\n", J->size2);


    /************************************/
    /* Centering matrix: I - 1/n * 11' */
    /***********************************/

    // Identity matrix    
    J = gsl_matrix_alloc(ROWS, ROWS);  
    gsl_matrix_set_identity(J);

    // Scale by -1/ROWS
    scale = gsl_matrix_alloc(ROWS, ROWS);
    gsl_matrix_set_all(scale, -1.0/ROWS);
    
    // Add matrices --> result is stored in J
    gsl_matrix_add(J, scale);                   
    gsl_matrix_free(scale);

    // gsl_matrix_fprintf(stdout, J, "%.4f");
    // printf("Rows J: %zu\n", J->size1);
    // printf("Columns J: %zu\n", J->size2);


    /************************************/
    /* Double center matrix B = -.5*JPJ */
    /************************************/

    // Allocate memory
    B = gsl_matrix_alloc(ROWS, ROWS);
    PJ = gsl_matrix_alloc(ROWS, ROWS);

    // PJ
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, P, J, 0.0, PJ);
    gsl_matrix_free(P);

    // -0.5*J
    gsl_matrix_scale(J, -0.5);    

    // Calculate B
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, J, PJ, 0.0, B);
    gsl_matrix_free(J);
    
    // gsl_matrix_fprintf(stdout, B, "%.4f");
    // printf("Rows B: %zu\n", B->size1);
    // printf("Columns B: %zu\n", B->size2);


    /**************************************/
    /* Top k Eigenvalues and Eigenvectors */
    /**************************************/

    // Allocate memory
    E = gsl_matrix_calloc(ROWS, ROWS);
    evals = gsl_vector_calloc(ROWS);
    w = gsl_eigen_symmv_alloc(ROWS);

    // Compute eigenvalues and eigenvectors and sort descending
    gsl_eigen_symmv(B, evals, E, w);
    gsl_eigen_symmv_free(w);
    gsl_eigen_symmv_sort(evals, E, GSL_EIGEN_SORT_VAL_DESC);

    // Select top three eigenvalues and their corresponding eigenvectors
    gsl_matrix_view E_topk = gsl_matrix_submatrix(E, 0, 0, ROWS, k);
    gsl_vector_view evals_topk = gsl_vector_subvector(evals, 0, k);
    
    // printf("\nAll eigenvalues:\n");
    // gsl_vector_fprintf(stdout, evals, "%.4f");
    // printf("\nTop k eigenvalues:\n");
    // gsl_vector_fprintf(stdout, &evals_topk.vector, "%.4f");
    // printf("\nAll eigenvectors:\n");
    // gsl_matrix_fprintf(stdout, E, "%.4f");
    // printf("\nTop k eigenvectors:\n");
    // gsl_matrix_fprintf(stdout, &E_topk.matrix, "%.4f");

    // Create diagonal matrix for sqrt(eigenvalues)
    L = gsl_matrix_calloc(k, k);
    for(int i=0; i<k; i++){
        gsl_matrix_set(L, i, i, sqrt(gsl_vector_get(&evals_topk.vector, i)));
    }
    // printf("Diagonal matrix: Sqrt of top k eigenvalues:\n");
    // gsl_matrix_fprintf(stdout, L, "%.4f");


    /****************************/
    /* Dimensionality Reduction */
    /****************************/

    // Reduce dimensions X_t = E_topk * L
    X_t = gsl_matrix_alloc(ROWS, k);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &E_topk.matrix, L, 0.0, X_t);
    
    // gsl_matrix_fprintf(stdout, X_t, "%.4f");
    // printf("Rows X_t: %zu\n", X_t->size1);
    // printf("Columns X_t: %zu\n", X_t->size2);

    // Free memory
    gsl_matrix_free(E);
    gsl_vector_free(evals);
    gsl_matrix_free(L);

    // Write matrix to disk
    ff = fopen("data_mds.txt", "w");
    gsl_matrix_fwrite(ff, X_t);     // Write matrix to binary file
    fclose(ff);

    gsl_matrix_free(X_t);
    printf("MDS Finished\n");

	return 0;
}


// Function definitions
double squared_l2_norm(gsl_vector *x, gsl_vector *y, int p){
    double result = 0.0;
    for(int j=0; j<p; j++){
        result += square(gsl_vector_get(x, j) - gsl_vector_get(y, j));
    }
    return result;
}