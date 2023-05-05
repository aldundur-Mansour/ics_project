#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>



// Function to allocate memory for a matrix of size n x n
long **createMatrix(int n) {
    long **matrix = (long **) malloc(n * sizeof(long *));
    for (int i = 0; i < n; i++) {
        matrix[i] = (long *) malloc(n * sizeof(long));
    }
    return matrix;
}

// Function to free memory allocated for a matrix
void freeMatrix(long **matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Function to read input matrices from a file
void readMatrices(char *filename, long **matrix1, long **matrix2, int n) {
    FILE *fp = fopen(filename, "r");
    fscanf(fp, "%d", &n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fscanf(fp, "%ld", &matrix1[i][j]);
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fscanf(fp, "%ld", &matrix2[i][j]);
        }
    }
    fclose(fp);
}

// Function to write output matrix to a file
void writeOutput(char *filename, long **matrix, int n) {
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "%d\n", n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(fp, "%ld ", matrix[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

// Function to write time taken by the algorithm to a file
void writeTime(char *filename, double time) {
    printf("%f\n", time);
    FILE *fp = fopen(filename, "w");
    int hours = (int) (time / 3600);
    int minutes = (int) ((time - hours * 3600) / 60);
    int seconds = (int) time % 60;
    int milliseconds = (int) ((time - (int) time) * 1000);
    fprintf(fp, "%02d:%02d:%02d.%03d\n", hours, minutes, seconds, milliseconds);
    fclose(fp);
}


void addMatrices(long **matrix1, long **matrix2, long **result, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = matrix1[i][j] + matrix2[i][j];
        }
    }
}

void subtractMatrices(long **matrix1, long **matrix2, long **result, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }
}

// Function to perform matrix multiplication using the straightforward divide and conquer algorithm
void multiplyMatricesSequential(long **matrix1, long **matrix2, long **result, int n, int threshold) {
    if (n <= threshold) {
//         Base case: perform matrix multiplication using standard algorithm
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = 0;
                for (int k = 0; k < n; k++) {
                    result[i][j] += (matrix1[i][k] * matrix2[k][j]);
                }
            }
        }

    } else {
        // Recursive case: divide matrices into four quadrants and recursively multiply them
        int m = n / 2;
        long **a11 = createMatrix(m);
        long **a12 = createMatrix(m);
        long **a21 = createMatrix(m);
        long **a22 = createMatrix(m);
        long **b11 = createMatrix(m);
        long **b12 = createMatrix(m);
        long **b21 = createMatrix(m);
        long **b22 = createMatrix(m);


        // Divide matrix1 into 4 quadrants
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                a11[i][j] = matrix1[i][j];
                a12[i][j] = matrix1[i][j + m];
                a21[i][j] = matrix1[i + m][j];
                a22[i][j] = matrix1[i + m][j + m];
            }
        }
        // Divide matrix2 into 4 quadrants
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                b11[i][j] = matrix2[i][j];
                b12[i][j] = matrix2[i][j + m];
                b21[i][j] = matrix2[i + m][j];
                b22[i][j] = matrix2[i + m][j + m];
            }
        }

        // Recursively multiply the quadrants
        long **c11 = createMatrix(m);
        long **c12 = createMatrix(m);
        long **c21 = createMatrix(m);
        long **c22 = createMatrix(m);
        multiplyMatricesSequential(a11, b11, c11, m, threshold);
        multiplyMatricesSequential(a12, b21, c12, m, threshold);
        multiplyMatricesSequential(a11, b12, c21, m, threshold);
        multiplyMatricesSequential(a12, b22, c22, m, threshold);

        // Combine the results to form the resulting matrix
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                result[i][j] = c11[i][j] + c12[i][j];
                result[i][j + m] = c21[i][j] + c22[i][j];
                result[i + m][j] = c11[i + m][j] + c12[i + m][j];
                result[i + m][j + m] = c21[i + m][j] + c22[i + m][j];
            }
        }
        freeMatrix(c11, m);
        freeMatrix(c12, m);
        freeMatrix(c21, m);
        freeMatrix(c22, m);
        freeMatrix(a11, m);
        freeMatrix(a12, m);
        freeMatrix(a21, m);
        freeMatrix(a22, m);
        freeMatrix(b11, m);
        freeMatrix(b12, m);
        freeMatrix(b21, m);
        freeMatrix(b22, m);
    }
}

void multiplyMatricesParallel(long **matrix1, long **matrix2, long **result, int n, int threshold) {
    if (n <= threshold) {
        multiplyMatricesSequential(matrix1, matrix2, result, n, threshold);
        return;
    }

    // Split matrices into quadrants
    int subSize = n / 2;
    long **a11 = createMatrix(subSize);
    long **a12 = createMatrix(subSize);
    long **a21 = createMatrix(subSize);
    long **a22 = createMatrix(subSize);
    long **b11 = createMatrix(subSize);
    long **b12 = createMatrix(subSize);
    long **b21 = createMatrix(subSize);
    long **b22 = createMatrix(subSize);
    long **c11 = createMatrix(subSize);
    long **c12 = createMatrix(subSize);
    long **c21 = createMatrix(subSize);
    long **c22 = createMatrix(subSize);
    long **p1 = createMatrix(subSize);
    long **p2 = createMatrix(subSize);
    long **p3 = createMatrix(subSize);
    long **p4 = createMatrix(subSize);
    long **p5 = createMatrix(subSize);
    long **p6 = createMatrix(subSize);
    long **p7 = createMatrix(subSize);
    long **temp1 = createMatrix(subSize);
    long **temp2 = createMatrix(subSize);

    // Split input matrices into quadrants
    for (int i = 0; i < subSize; i++) {
        for (int j = 0; j < subSize; j++) {
            a11[i][j] = matrix1[i][j];
            a12[i][j] = matrix1[i][j + subSize];
            a21[i][j] = matrix1[i + subSize][j];
            a22[i][j] = matrix1[i + subSize][j + subSize];
            b11[i][j] = matrix2[i][j];
            b12[i][j] = matrix2[i][j + subSize];
            b21[i][j] = matrix2[i + subSize][j];
            b22[i][j] = matrix2[i + subSize][j + subSize];
        }
    }

    // Calculate p1 to p7 in parallel
#pragma omp parallel sections
    {
#pragma omp section
        {
            // p1 = a11 * (b12 - b22)
            subtractMatrices(b12, b22, temp1, subSize);
            multiplyMatricesParallel(a11, temp1, p1, subSize, threshold);
        }
#pragma omp section
        {
            // p2 = (a11 + a12) * b22
            addMatrices(a11, a12, temp1, subSize);
            multiplyMatricesParallel(temp1, b22, p2, subSize, threshold);
        }
#pragma omp section
        {
            // p3 = (a21 + a22) * b11
            addMatrices(a21, a22, temp1, subSize);
            multiplyMatricesParallel(temp1, b11, p3, subSize, threshold);
        }
#pragma omp section
        {
            // p4 = a22 * (b21 - b11)
            subtractMatrices(b21, b11, temp1, subSize);
            multiplyMatricesParallel(a22, temp1, p4, subSize, threshold);
        }
#pragma omp section
        {
// p5 = (a11 + a22) * (b11 + b22)
            addMatrices(a11, a22, temp1, subSize);
            addMatrices(b11, b22, temp2, subSize);
            multiplyMatricesParallel(temp1, temp2, p5, subSize, threshold);
        }
#pragma omp section
        {
// p6 = (a12 - a22) * (b21 + b22)
            subtractMatrices(a12, a22, temp1, subSize);
            addMatrices(b21, b22, temp2, subSize);
            multiplyMatricesParallel(temp1, temp2, p6, subSize, threshold);
        }
#pragma omp section
        {
// p7 = (a11 - a21) * (b11 + b12)
            subtractMatrices(a11, a21, temp1, subSize);
            addMatrices(b11, b12, temp2, subSize);
            multiplyMatricesParallel(temp1, temp2, p7, subSize, threshold);
        }
    }

// Calculate result matrices in parallel
#pragma omp parallel sections
    {
#pragma omp section
        {
            // c11 = p5 + p4 - p2 + p6
            addMatrices(p5, p4, temp1, subSize);
            subtractMatrices(temp1, p2, temp2, subSize);
            addMatrices(temp2, p6, c11, subSize);
        }
#pragma omp section
        {
            // c12 = p1 + p2
            addMatrices(p1, p2, c12, subSize);
        }
#pragma omp section
        {
            // c21 = p3 + p4
            addMatrices(p3, p4, c21, subSize);
        }
#pragma omp section
        {
            // c22 = p5 + p1 - p3 - p7
            addMatrices(p5, p1, temp1, subSize);
            subtractMatrices(temp1, p3, temp2, subSize);
            subtractMatrices(temp2, p7, c22, subSize);
        }
    }

// Combine result matrices into one
    for (int i = 0; i < subSize; i++) {
        for (int j = 0; j < subSize; j++) {
            result[i][j] = c11[i][j];
            result[i][j + subSize] = c12[i][j];
            result[i + subSize][j] = c21[i][j];
            result[i + subSize][j + subSize] = c22[i][j];
        }
    }

// Free dynamically allocated memory
    freeMatrix(a11, subSize);
    freeMatrix(a12, subSize);
    freeMatrix(a21, subSize);
    freeMatrix(a22, subSize);
    freeMatrix(b11, subSize);
    freeMatrix(b12, subSize);
    freeMatrix(b21, subSize);
    freeMatrix(b22, subSize);
    freeMatrix(c11, subSize);
    freeMatrix(c12, subSize);
    freeMatrix(c21, subSize);
    freeMatrix(c22, subSize);
    freeMatrix(p1, subSize);
    freeMatrix(p2, subSize);
    freeMatrix(p3, subSize);
    freeMatrix(p4, subSize);
    freeMatrix(p5, subSize);
    freeMatrix(p6, subSize);
    freeMatrix(p7, subSize);
    freeMatrix(temp1, subSize);
    freeMatrix(temp2, subSize);

}

void strassenMatricesSequential(long **matrix1, long **matrix2, long **result, int n, int threshold) {
    if (n <= threshold) {
        // Base case: perform matrix multiplication using standard algorithm
        multiplyMatricesSequential(matrix1, matrix2, result, n, threshold);
        return;
    } else {
        // Recursive case: divide matrices into four quadrants and recursively multiply them
        int m = n / 2;

        long **a11 = createMatrix(m);
        long **a12 = createMatrix(m);
        long **a21 = createMatrix(m);
        long **a22 = createMatrix(m);

        long **b11 = createMatrix(m);
        long **b12 = createMatrix(m);
        long **b21 = createMatrix(m);
        long **b22 = createMatrix(m);

        long **c11 = createMatrix(m);
        long **c12 = createMatrix(m);
        long **c21 = createMatrix(m);
        long **c22 = createMatrix(m);

        long **p1 = createMatrix(m);
        long **p2 = createMatrix(m);
        long **p3 = createMatrix(m);
        long **p4 = createMatrix(m);
        long **p5 = createMatrix(m);
        long **p6 = createMatrix(m);
        long **p7 = createMatrix(m);

        // Divide matrix1 into 4 quadrants
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                a11[i][j] = matrix1[i][j];
                a12[i][j] = matrix1[i][j + m];
                a21[i][j] = matrix1[i + m][j];
                a22[i][j] = matrix1[i + m][j + m];
            }
        }

        // Divide matrix2 into 4 quadrants
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                b11[i][j] = matrix2[i][j];
                b12[i][j] = matrix2[i][j + m];
                b21[i][j] = matrix2[i + m][j];
                b22[i][j] = matrix2[i + m][j + m];
            }
        }

        // Compute the 7 products
        subtractMatrices(b12, b22, c11, m);
        strassenMatricesSequential(a11, c11, p1, m, threshold);

        addMatrices(a11, a12, c11, m);
        strassenMatricesSequential(c11, b22, p2, m, threshold);

        addMatrices(a21, a22, c11, m);
        strassenMatricesSequential(c11, b11, p3, m, threshold);

        subtractMatrices(b21, b11, c11, m);
        strassenMatricesSequential(a22, c11, p4, m, threshold);

        addMatrices(a11, a22, c11, m);
        addMatrices(b11, b22, c12, m);
        strassenMatricesSequential(c11, c12, p5, m, threshold);

        subtractMatrices(a12, a22, c11, m);
        addMatrices(b21, b22, c12, m);
        strassenMatricesSequential(c11, c12, p6, m, threshold);

        subtractMatrices(a11, a21, c11, m);
        addMatrices(b11, b12, c12, m);
        strassenMatricesSequential(c11, c12, p7, m, threshold);

        // Compute the four quadrants of the result matrix
        addMatrices(p5, p4, c11, m);
        subtractMatrices(c11, p2, c11, m);
        addMatrices(c11, p6, c11, m);

        addMatrices(p1, p2, c12, m);

        addMatrices(p3, p4, c21, m);

        subtractMatrices(p5, p1, c22, m);
        addMatrices(c22, p3, c22, m);
        addMatrices(c22, p7, c22, m);

        // Combine the four quadrants into the final result matrix
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < m; j++) {
                result[i][j] = c11[i][j];
                result[i][j + m] = c12[i][j];
                result[i + m][j] = c21[i][j];
                result[i + m][j + m] = c22[i][j];
            }
        }

        // Deallocate memory
        freeMatrix(a11, m);
        freeMatrix(a12, m);
        freeMatrix(a21, m);
        freeMatrix(a22, m);

        freeMatrix(b11, m);
        freeMatrix(b12, m);
        freeMatrix(b21, m);
        freeMatrix(b22, m);

        freeMatrix(c11, m);
        freeMatrix(c12, m);
        freeMatrix(c21, m);
        freeMatrix(c22, m);

        freeMatrix(p1, m);
        freeMatrix(p2, m);
        freeMatrix(p3, m);
        freeMatrix(p4, m);
        freeMatrix(p5, m);
        freeMatrix(p6, m);
        freeMatrix(p7, m);
    }
}

void strassenMatricesParallel(long **matrix1, long **matrix2, long **result, int n, int threshold) {
    if (n <= threshold) {
        // Base case: perform matrix multiplication using standard algorithm
        multiplyMatricesParallel(matrix1, matrix2, result, n, threshold);
        return;
    } else {
        // Recursive case: divide matrices into four quadrants and recursively multiply them
        int m = n / 2;

        long **a11 = createMatrix(m);
        long **a12 = createMatrix(m);
        long **a21 = createMatrix(m);
        long **a22 = createMatrix(m);

        long **b11 = createMatrix(m);
        long **b12 = createMatrix(m);
        long **b21 = createMatrix(m);
        long **b22 = createMatrix(m);

        long **c11 = createMatrix(m);
        long **c12 = createMatrix(m);
        long **c21 = createMatrix(m);
        long **c22 = createMatrix(m);

        long **p1 = createMatrix(m);
        long **p2 = createMatrix(m);
        long **p3 = createMatrix(m);
        long **p4 = createMatrix(m);
        long **p5 = createMatrix(m);
        long **p6 = createMatrix(m);
        long **p7 = createMatrix(m);

        // Divide matrix1 into 4 quadrants
#pragma omp parallel sections num_threads(8)
        {
#pragma omp section 
{
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    a11[i][j] = matrix1[i][j];
                }
            }
}
#pragma omp section
{
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    a12[i][j] = matrix1[i][j + m];
                }
            }
}
#pragma omp section
{
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    a21[i][j] = matrix1[i + m][j];
                }
            }
}
#pragma omp section
{
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    a22[i][j] = matrix1[i + m][j + m];
                }
            }
        }
        }
        // Divide matrix2 into 4 quadrants
#pragma omp parallel sections num_threads(8)
        {
#pragma omp section
{
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    b11[i][j] = matrix2[i][j];
                }
            }
}
#pragma omp section
{
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    b12[i][j] = matrix2[i][j + m];
                }
            }
}
#pragma omp section
{
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    b21[i][j] = matrix2[i + m][j];
                }
            }
}
#pragma omp section
{
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    b22[i][j] = matrix2[i + m][j + m];
                }
            }
        }
        }

        // Compute seven products recursively
#pragma omp parallel sections num_threads(8)
{
#pragma omp section
{
//             // Compute the 7 products
//        subtractMatrices(b12, b22, c11, m);
//        strassenMatricesSequential(a11, c11, p1, m, threshold);
//
//        addMatrices(a11, a12, c11, m);
//        strassenMatricesSequential(c11, b22, p2, m, threshold);
            subtractMatrices(b12, b22, c11, m);
            strassenMatricesParallel(a11, c11, p1, m, threshold);
}

#pragma omp section
{
            addMatrices(a11, a12, c11, m);
            strassenMatricesParallel(c11, b22, p2, m, threshold);
}
#pragma omp section
{
            addMatrices(a21, a22, c11, m);
            strassenMatricesParallel(c11, b11, p3, m, threshold);
}
#pragma omp section
{
            subtractMatrices(b21, b11, c11, m);
            strassenMatricesParallel(a22, c11, p4, m, threshold);
}
#pragma omp section
{
            addMatrices(a11, a22, c11, m);
            addMatrices(b11, b22, c12, m);
            strassenMatricesParallel(c11, c12, p5, m, threshold);
}
#pragma omp section
{
            subtractMatrices(a12, a22, c11, m);
            addMatrices(b21, b22, c12, m);
            strassenMatricesParallel(c11, c12, p6, m, threshold);
}
#pragma omp section
{
            subtractMatrices(a11, a21, c11, m);
            addMatrices(b11, b12, c12, m);
            strassenMatricesParallel(c11, c12, p7, m, threshold);
}
        }



        // Compute result matrices from the products
#pragma omp parallel sections num_threads(8)
        {
#pragma omp section 
{
            addMatrices(p5, p4, c11, m);
            subtractMatrices(c11, p2, c11, m);
            addMatrices(c11, p6, c11, m);
}
#pragma omp section
{
            addMatrices(p3, p5, c12, m);
}
#pragma omp section
{
            addMatrices(p2, p4, c21, m);
}
#pragma omp section
{
            subtractMatrices(p5, p1, c22, m);
            addMatrices(c22, p3, c22, m);
            addMatrices(c22, p7, c22, m);
}
        }

        // Combine result matrices into a single matrix
#pragma omp parallel sections num_threads(8)
        {
#pragma omp section
{
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    result[i][j] = c11[i][j];
                }
            }
}
#pragma omp section
{
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    result[i][j + m] = c12[i][j];
                }
            }
}
#pragma omp section
{
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    result[i + m][j] = c21[i][j];
                }
            }
}
#pragma omp section
{
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < m; j++) {
                    result[i + m][j + m] = c22[i][j];
                }
            }
        }
        }
        // Free memory
        freeMatrix(a11, m);
        freeMatrix(a12, m);
        freeMatrix(a21, m);
        freeMatrix(a22, m);

        freeMatrix(b11, m);
        freeMatrix(b12, m);
        freeMatrix(b21, m);
        freeMatrix(b22, m);

        freeMatrix(c11, m);
        freeMatrix(c12, m);
        freeMatrix(c21, m);
        freeMatrix(c22, m);

        freeMatrix(p1, m);
        freeMatrix(p2, m);
        freeMatrix(p3, m);
        freeMatrix(p4, m);

        freeMatrix(p5, m);
        freeMatrix(p6, m);
        freeMatrix(p7, m);

    }
}

void writeMatrices1(char *filename, long **matrix1, long **matrix2, int n) {
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "%d\n", n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(fp, "%ld ", matrix1[i][j]);
        }
        fprintf(fp, "\n");
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(fp, "%ld ", matrix2[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

int main() {
    // Initialize variables
    int n = 4096 ; // Size of matrices
    int threshold = 128; // Threshold value for switching to sequential algorithm
    // Create matrices
    printf("Creating matrices of size %d\n", n);
    long **matrix1 = createMatrix(n);
    long **matrix2 = createMatrix(n);
    long **result1 = createMatrix(n); // Result of sequential multiplication
    long **result2 = createMatrix(n); // Result of parallel multiplication
    long **result3 = createMatrix(n); // Result of sequential Strassen's algorithm
    long **result4 = createMatrix(n); // Result of parallel Strassen's algorithm

    // Fill matrices with random data
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix1[i][j] = rand() % 10;
            matrix2[i][j] = rand() % 10;
        }
    }

    // Write input matrices to file
    writeMatrices1("general_matrix.txt", matrix1, matrix2, n);

    // Read input matrices from file
    printf("Reading input matrices from file\n");
    readMatrices("general_matrix.txt", matrix1, matrix2, n);

    if(threshold < n) {
        threshold = n;
    }

    // Multiply matrices using sequential algorithm and time it
    printf("Multiplying matrices using sequential algorithm\n");
    clock_t start = clock();
    multiplyMatricesSequential(matrix1, matrix2, result1, n, threshold);
    double sequentialTime = (double) (clock() - start) / CLOCKS_PER_SEC;

    // Write output matrix to file
    printf("Writing output matrix to file\n");
    writeOutput("general_matrix_500_output_StraightDivAndConq.txt", result1, n);
    writeTime("general_matrix_500_info_StraightDivAndConq.txt", sequentialTime);

    // Multiply matrices using parallel algorithm and time it
    printf("Multiplying matrices using parallel algorithm\n");
    start = clock();
    multiplyMatricesParallel(matrix1, matrix2, result2, n, threshold);
    double parallelTime = (double) (clock() - start) / CLOCKS_PER_SEC;

    // Write output matrix to file
    printf("Writing output matrix to file\n");
    writeOutput("general_matrix_500_output_StraightDivAndConqP.txt", result2, n);
    writeTime("general_matrix_500_info_StraightDivAndConqP.txt", parallelTime);

    // Multiply matrices using sequential Strassen's algorithm and time it
    printf("Multiplying matrices using sequential Strassen's algorithm\n");
    start = clock();
    strassenMatricesSequential(matrix1, matrix2, result3, n, threshold);
    double sequentialStrassenTime = (double) (clock() - start) / CLOCKS_PER_SEC;

    // Write output matrix to file
    printf("Writing output matrix to file\n");
    writeOutput("general_matrix_500_output_StrassenDivAndConq.txt", result3, n);
    writeTime("general_matrix_500_info_StrassenDivAndConq.txt", sequentialStrassenTime);

    // Multiply matrices using parallel Strassen's algorithm and time it
    printf("Multiplying matrices using parallel Strassen's algorithm\n");
    start = clock();
    strassenMatricesParallel(matrix1, matrix2, result4, n, threshold);
    double parallelStrassenTime = (double) (clock() - start) / CLOCKS_PER_SEC;

    // Write output matrix to file
    printf("Writing output matrix to file\n");
    writeOutput("general_matrix_500_output_StrassenDivAndConqP.txt", result4, n);
    writeTime("general_matrix_500_info_StrassenDivAndConqP.txt", parallelStrassenTime);

    // Check that results are the same
    int isEqual = 1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (result1[i][j] != result2[i][j] || result1[i][j] != result3[i][j] || result1[i][j] != result4[i][j]) {
                isEqual = 0;
                break;
            }
        }
        if (!isEqual) {
            break;
        }
    }
    if (isEqual) {
        printf("Results are equal\n");
    } else {
        printf("Results are not equal\n");
    }
// Print execution times
    printf("Sequential algorithm execution time: %lf seconds\n", sequentialTime);
    printf("Parallel algorithm execution time: %lf seconds\n", parallelTime);
    printf("Sequential Strassen's algorithm execution time: %lf seconds\n", sequentialStrassenTime);
    printf("Parallel Strassen's algorithm execution time: %lf seconds\n", parallelStrassenTime);

// Free memory
    freeMatrix(matrix1, n);
    freeMatrix(matrix2, n);
    freeMatrix(result1, n);
    freeMatrix(result2, n);
    freeMatrix(result3, n);
    freeMatrix(result4, n);

    return 0;
}


