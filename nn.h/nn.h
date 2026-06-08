// Understand this file first when coming in next time !!!

#ifndef NN_H_ // why is it called inclusion guard??
#define NN_H_

#include<stddef.h>
#include<stdio.h>

#ifndef NN_MALLOC
#include<stdlib.h>
#define NN_MALLOC malloc
#endif

#ifndef NN_ASSERT
#include<assert.h>
#define NN_ASSERT assert
#endif

#define MAT_AT(m, i, j) (m).es[(i) * (m).cols + (j)]

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
void mat_print(Mat a);
void mat_rand(Mat a, float low, float high);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

float float_rand(float low, float high){
    return ((float) rand() / (float) RAND_MAX) * (high - low) + low;
}

Mat mat_alloc(size_t rows, size_t cols){
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.es = NN_MALLOC(sizeof(*(m.es)) * rows * cols);
    NN_ASSERT(m.es != NULL);
    return m;
}

void mat_print(Mat a){
    for(size_t i= 0; i< a.rows; i++){
        for(size_t j= 0; j < a.cols; j++){
            printf("%f\t", MAT_AT(a, i, j));
        }
        printf("\n");
    }
}

void mat_rand(Mat a, float low, float high){

    for(size_t i= 0; i< a.rows; i++){
        for(size_t j= 0; j< a.cols; j++){
            MAT_AT(a, i, j) = float_rand(-10.0f, 10.0f);
        }
    }
}

void mat_dot(Mat dst, Mat a, Mat b){
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);

    for(size_t i= 0; i< dst.rows){
        for(size_t j= 0; j< dst.cols; j++){
            // TRY TO IMPLEMENT ON YOUR OWN BEFORE WATCHING THE VIDEO
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

