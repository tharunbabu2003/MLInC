#include<time.h>

#define NN_IMPLEMENTATION // Without this, nn.h acts like a header. With this nn.h acts like a (header + C file). Why? I need to explore later...
#include "nn.h"

int main(void){
    srand((unsigned int) time(NULL));

    Mat m;
    m = mat_alloc(1, 2);
    mat_rand(m, -10.0f, 10.0f);
    mat_print(m);
    
    return 0;
}