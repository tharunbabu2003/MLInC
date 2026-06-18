// Each layer is a matrix, each column is a neuron

#ifndef TINY_ML_
#define TINY_ML_

#ifndef NN_MALLOC
#include<stdlib.h>
#define NN_MALLOC malloc
#endif

#ifndef NN_ASSERT
#include<assert.h>
#define NN_ASSERT assert
#endif

#define MAT_AT(m, i, j) (m).es[(i) * (m).cols + (j)]
#define MAT_PRINT(m) mat_print(m, #m);

typedef struct
{
    size_t rows;
    size_t cols;
    // int stride;
    float *es;
} Mat;

Mat mat_alloc(size_t rows, size_t cols);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_add(Mat dst, Mat a);
void mat_print(Mat a, char* mname);
void mat_rand(Mat a, float low, float high);

#endif // TINY_ML_

#ifdef NN_IMPLEMENTATION
#include<stdio.h>

float float_rand(float low, float high){
    return ((float) rand() / (float) RAND_MAX) * (high - low) + low;
}

Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.rows = rows;
    m.cols= cols;
    m.es = NN_MALLOC(sizeof(*(m.es)) * rows * cols);
    NN_ASSERT(m.es != NULL);
    return m;
}

void mat_print(Mat a, char* mat_name){
    printf("%s = [\n", mat_name);
    for(size_t i= 0; i< a.rows; i++){
        for(size_t j= 0; j < a.cols; j++){
            printf("%f\t", MAT_AT(a, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}

void mat_rand(Mat a, float low, float high){

    for(size_t i= 0; i< a.rows; i++){
        for(size_t j= 0; j< a.cols; j++){
            MAT_AT(a, i, j) = float_rand(low, high);
        }
    }
}

/*
float fn(Mat a, Mat b, size_t rn, size_t cn){ // rn = row number, cn= column number
    float output= 0.0f;
    for(size_t i= 0; i< a.cols; i++){
        output += MAT_AT(a, rn, i) * MAT_AT(b, i, cn);
    }
    return output;
}
*/

void mat_dot(Mat dst, Mat a, Mat b){
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);

    for(size_t i= 0; i< dst.rows; i++){
        for(size_t j= 0; j< dst.cols; j++){
            MAT_AT(dst, i, j)= 0;
            for(size_t k= 0; k< a.cols; k++){
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

void mat_add(Mat dst, Mat a){
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);
    for(size_t i=0; i < dst.rows; i++){
        for(size_t j= 0; j< dst.cols; j++){
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

#endif // NN_IMPLEMENTATION

