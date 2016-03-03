
/*********************************************************************************/
/* Given m sets of data points (Xi, Yi), represented by matrix A and a vector B, */
/* this procedure performs the Least-Square Fitting using the formula:           */
/*             transpose(A)*A*X = transpose(A)*B,   and solves for X.            */
/* Provided by Jin-Long Chen, January 1991                   */
/* Modified by George Stockman, October 1997 (generalized)                       */
/*********************************************************************************/


#include <stdio.h>
#include <math.h>

#define abs(x) ((x > 0) ? (x) : (-x))
#define max(x,y) ((x > y) ? (x) : (y))
#define ERROR 0
#define MaxVars 20
#define MaxEqs 100

int decomp(const int n, float A[][MaxVars], float B[], float X[])
{
    int pivot[MaxVars];
    float rowmax, colmax, D[MaxVars], Y[MaxVars], temp, scale, ratio, sum;
    int i, j, k, istar, t, ip;

    for (i = 0; i < n; i++)
    {
        pivot[i] = i;
        rowmax = 0.0;

        for (j = 0; j < n; j++)
            rowmax = max(rowmax, abs(A[i][j]));

        D[i] = rowmax;
        if (rowmax == 0.0)
        {
            // printf("Singular matrix in routine decomp()\n");
            return(ERROR);
        }
    }

    for (k = 0; k < n - 1; k++)
    {
        colmax = abs(A[k][k]) / D[k];
        istar = k;

        /* the maximum of scaled partial pivoting is picked */
        for (i = k+1; i < n; i++)
        {
            scale = abs(A[i][k]) / D[i];
            if (scale > colmax)
            {
                colmax = scale;
                istar = i;
            }
        }
        if (colmax == 0.0)
        {
            // printf("Singular matrix in routine decomp()\n");
            return(ERROR);
        }

        if (istar > k)
        {
            /* exchange the index of pivot */
            t = pivot[istar];
            pivot[istar] = pivot[k];
            pivot[k] = t;

            /* exchange the row maximum accordingly */
            temp = D[istar];
            D[istar] = D[k];
            D[k] = temp;

            /* exchange two rows */
            for (j = 0; j < n; j++)
            {
                temp = A[istar][j];
                A[istar][j] = A[k][j];
                A[k][j] = temp;
            }
        }

        /******************************************************************/
        /* A(i,j) stores the lower-triangular matrix L(i,j), where i > j  */
        /* A(i,j) stores the upper-triangular matrix U(i,j), where i <= j */
        /******************************************************************/
        for (i = k + 1; i < n; i++)
        {
            A[i][k] = A[i][k] / A[k][k];
            ratio = A[i][k];

            for (j = k + 1; j < n; j++)
                A[i][j] = A[i][j] - ratio*A[k][j];
        }

    }

    if (A[n-1][n-1] == 0.0)
    {
        // printf("Singular matrix in routine decomp()\n");
        return(ERROR);
    }

    /****************************************************************/
    /* matrix A now contains LU decomposition. Forward substitution */
    /* solves L*y = B, then backward substitution solves U*x = y.   */
    /****************************************************************/

    /* Forward Substitution */
    for (i = 0; i < n; i++)
    {
        ip = pivot[i];
        sum = 0.0;

        for (j = 0; j <= i - 1; j++)
            sum += A[i][j]*Y[j];

        Y[i] = B[ip] - sum;
    }

    /* Back Substitution */
    for (i = n-1; i >= 0; i--)
    {
        sum = 0.0;

        for (j = i + 1; j < n; j++)
            sum += A[i][j]*X[j];
        X[i] = (Y[i] - sum) / A[i][i];
    }

    return(1);
}
/********* end of code decomp***********/


int LeastSquaresFit( const int Neqs, const int Nvars, float (*A)[MaxVars], float B[], float X[] )
{
    /* See comments in the book Image Warping by George Wolberg, page 65 and
    the cited readings, regarding precision of the technique.          */

    int i, j, k;
    float sum, TA[MaxVars][MaxVars], TB[MaxVars];
    int ix;

    /* TB = Transpose(A) * B */
    for (j = 0; j < Nvars; ++j)
    {
        TB[j] = 0.0;
        for (k = 0; k < Neqs; ++k)
        {
            TB[j] += A[k][j] * B[k];
        }
    }

    /*  TA = Transpose(A)*A  */
    for (i = 0; i < Nvars; ++i)
    {
        for (j = 0; j <= i; ++j)
        {
            TA[i][j] = 0.0;
            for (k = 0; k < Neqs; ++k)
            {
                TA[i][j] += A[k][i] * A[k][j];
            }
            TA[j][i] = TA[i][j];
        }
    }

    /*  solve TA*X = TB  */
    if (decomp(Nvars, TA, TB, X) == ERROR)
        return(ERROR);

    /* compute the residual E = A*X - B */
    for (k = 0; k < Neqs; k++)
    {
        sum = 0.0;
        for (j = 0; j < Nvars; j++)
            sum += A[k][j]*X[j];

        B[k] = sum - B[k];   /* store the residual in vector B */
    }

    return(1);
}

void printMatrixToFile(float* Hleft, float* Hright, char* filename)
{
    int i;
    FILE* fp = fopen(filename, "w+");

    fprintf(fp, "%f %f %f %f %f %f %f %f %f\n", Hleft[0], Hleft[1], Hleft[2],
            Hleft[3], Hleft[4], Hleft[5],
            Hleft[6], Hleft[7], Hleft[8]);

    fprintf(fp, "%f %f %f %f %f %f %f %f %f\n", Hright[0], Hright[1], Hright[2],
            Hright[3], Hright[4], Hright[5],
            Hright[6], Hright[7], Hright[8]);

    fclose(fp);
    printf("Done!");
}


float findAffine(int NcontrolPts,int **matches, float *X)
{
    // int   NcontrolPts = 0;
    int   Neqs=0, Nvars=6, i, j, eq, num;
    float A[MaxEqs][MaxVars], B[MaxEqs]; //X[MaxEqs];
    float identity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    X[6]=0; X[7]=0; X[8]=1;

    // Obtain 2 equations for each pair of matching control points
    // int xi,yi,ui,vi;   // control points  (xi, yi) and (ui, vi)
    char ans, filename[100];

    // printf("How many control point matches? (int) -\n");
    // scanf("%d", &NcontrolPts);

    if ( 2*NcontrolPts > MaxEqs )
    {
        printf("****Not enough memory****\n");
        printf("using only %d control points\n", MaxEqs/2);
        NcontrolPts = MaxEqs/2;
    }

    printf("Found %d control point pairs src_x src_y dst_x dst_y \n", NcontrolPts);

    Neqs = 0;
    for (i=0; i<NcontrolPts; i++)
    {
        // scanf("%d %d %d %d", &xi, &yi, &ui, &vi);
        eq = 2*i;
        A[eq][0]=matches[i][0];  A[eq][1]=matches[i][1];  A[eq][2]=1.0;
        A[eq][3]=0.0; A[eq][4]=0.0; A[eq][5]=0.0;
        B[eq]=matches[i][2];

        eq=eq+1;
        A[eq][0]=0.0;  A[eq][1]=0.0;  A[eq][2]=0.0;
        A[eq][3]=matches[i][0];  A[eq][4]=matches[i][1];  A[eq][5]=1.0;
        B[eq]=matches[i][3];

        Neqs=Neqs+2;
    }

    /* call the least squares routine to find the "best" set of */
    /* parameters from the  m  equations       */

    if (LeastSquaresFit(Neqs, Nvars, A, B, X) == ERROR)
    {
        printf("Error detected in Least-square function\n");
        return 0;
    }

    // printf("\nThe Transformation Matrix is: \n");
    // printf("\n[%f , %f , %f \n%f , %f , %f \n0.000000 0.000000 1.000000 ]\n", X[0], X[1], X[2], X[3] ,X[4], X[5]);

    // printf("Print left and right matrices to file? (y/n) :\n");
    // scanf(" %c", &ans);

    // if(ans == 'y' || ans == 'Y')
    // {
    //     printf("Enter filename: \n");
    //     scanf(" %s", filename);
    //     printMatrixToFile(identity, X, filename);
    // }


    // cout << "\n Residuals for " << Neqs << " equations are as follows:" ;
    // for (i=0; i<Neqs; i++)
    //   cout << "\n i= " << i << " error= " << B[i];
    // cout << "\n\n======Fitting Program Complete======\n\n";
    // return X;
}
