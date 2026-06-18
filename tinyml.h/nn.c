#include<time.h>

#define NN_IMPLEMENTATION // Without this, nn.h acts like a header. With this nn.h acts like a (header + C file). Why? I need to explore later...
#include "tinyml.h"

int main(void){
    // Implementing XOR model with tinyml framework

    Mat w1 = mat_alloc(2, 2);
    mat_rand(w1, 0, 1);

    Mat b1 = mat_alloc(1, 2);
    mat_rand(b1, 0, 1);

    Mat w2 = mat_alloc(2, 1);
    mat_rand(w2, 0, 1);

    Mat b2 = mat_alloc(1, 1);
    mat_rand(b2, 0, 1);

    MAT_PRINT(w1);
    MAT_PRINT(b1);
    MAT_PRINT(w2);
    MAT_PRINT(b2);
    return 0;
}