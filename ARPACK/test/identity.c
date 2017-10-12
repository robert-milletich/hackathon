#include"lapacke.h"
#include <stdio.h>

void av_identity_(const int *Nptr, const float *x, float *y)
{
/*
 * Computes  y = I * x
 */

  int i;

  for (i=0;i<*Nptr;++i)
	y[i] = x[i];
}

void av_csv_(const int *Nptr, const float *x, float *y)
{
//  FILE *fd = fopen();

/* read CSV here */

const float matrix[4][4]=
{
	{0, 1, 2, 3},
        {1, 0, 4, 5},
        {2, 4, 0, 6},
        {3, 5, 6, 0}

};

/* multiply CSV matrix by "x" and store it in "y" */

cblas_sgemv(101,111,4,4,1.0,matrix,4,x,1,0.0,y,1);

// fclose(fd);
}
