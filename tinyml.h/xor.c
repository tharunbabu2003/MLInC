#include<time.h>

#define NN_IMPLEMENTATION // Without this, nn.h acts like a header. With this nn.h acts like a (header + C file). Why? I need to explore later...
#include "tinyml.h"

typedef struct{
    Mat a0;
    Mat a1, w1, b1;
    Mat a2, w2, b2;
} Xor;

Xor xor_alloc(){
    Xor m;
    m.a0= mat_alloc(1, 2);
    int x1= 0, x2= 1;

    MAT_AT(m.a0, 0, 0) = x1;
    MAT_AT(m.a0, 0, 1) = x2;

    m.w1 = mat_alloc(2, 2);
    mat_rand(m.w1, 0, 1);

    m.b1 = mat_alloc(1, 2);
    mat_rand(m.b1, 0, 1);

    m.a1 = mat_alloc(1, 2);

    m.w2 = mat_alloc(2, 1);
    mat_rand(m.w2, 0, 1);

    m.b2 = mat_alloc(1, 1);
    mat_rand(m.b2, 0, 1);

    m.a2 = mat_alloc(1, 1);

    return m;
}

float forward(Xor m){

    mat_dot(m.a1, m.a0, m.w1);
    mat_add(m.a1, m.b1);
    activation(m.a1);

    mat_dot(m.a2, m.a1, m.w2);
    mat_add(m.a2, m.b2);
    activation(m.a2);

    return *m.a2.es;

}

float cost(Xor m, Mat ti, Mat to){
    NN_ASSERT(ti.rows == to.rows);
    NN_ASSERT(to.cols == m.a2.cols);

    size_t n= ti.rows;
    Mat y= mat_alloc(1, to.cols);
    float c= 0.0f;

    for(size_t i= 0; i< n; ++i){
        mat_copy(m.a0, mat_row(ti, i));
        mat_copy(y, mat_row(to, i));

        forward(m);

        size_t q= to.cols;
        for(size_t j= 0; j< q; ++j){
            float diff = MAT_AT(y, 0, j) - MAT_AT(m.a2, 0, j);
            c += diff * diff;
        }
    }
    return c/n;
}

void finite_diff(Xor m, Xor g, Mat ti, Mat to, float eps){
    float c= cost(m, ti, to);
    float saved;
    for(size_t i= 0; i< m.w1.rows; ++i){
        for(size_t j= 0; j< m.w1.cols; ++j){
            saved= MAT_AT(m.w1, i, j);
            MAT_AT(m.w1, i, j) += eps;
            MAT_AT(g.w1, i, j) = (cost(m, ti, to) - c)/eps;
            MAT_AT(m.w1, i, j) = saved;
        }
    }

    for(size_t i= 0; i< m.b1.rows; ++i){
        for(size_t j= 0; j< m.b1.cols; ++j){
            saved= MAT_AT(m.b1, i, j);
            MAT_AT(m.b1, i, j) += eps;
            MAT_AT(g.b1, i, j)= (cost(m, ti, to) - c)/ eps;
            MAT_AT(m.b1, i, j)= saved;
        }
    }

    for(size_t i= 0; i< m.w2.rows; ++i){
        for(size_t j= 0; j< m.w2.cols; ++j){
            saved= MAT_AT(m.w2, i, j);
            MAT_AT(m.w2, i, j) += eps;
            MAT_AT(g.w2, i, j)= (cost(m, ti, to) - c)/ eps;
            MAT_AT(m.w2, i, j)= saved;
        }
    }

    for(size_t i= 0; i< m.b2.rows; ++i){
        for(size_t j= 0; j< m.b2.cols; ++j){
            saved= MAT_AT(m.b2, i, j);
            MAT_AT(m.b2, i, j) += eps;
            MAT_AT(g.b2, i, j)= (cost(m, ti, to) - c)/ eps;
            MAT_AT(m.b2, i, j)= saved;
        }
    }
}

void update_weights(Xor m, Xor g, float lr){

    for(size_t i= 0; i< m.w1.rows; ++i){
        for(size_t j= 0; j< m.w1.cols; ++j){
            MAT_AT(m.w1, i, j) -= lr* MAT_AT(g.w1, i, j);
        }
    }

    for(size_t i= 0; i< m.b1.rows; ++i){
        for(size_t j= 0; j< m.b1.cols; ++j){
            MAT_AT(m.b1, i, j) -= lr* MAT_AT(g.b1, i, j);
        }
    }

    for(size_t i= 0; i< m.w2.rows; ++i){
        for(size_t j= 0; j< m.w2.cols; ++j){
            MAT_AT(m.w2, i, j) -= lr* MAT_AT(g.w2, i, j);
        }
    }

    for(size_t i= 0; i< m.b2.rows; ++i){
        for(size_t j= 0; j< m.b2.cols; ++j){
            MAT_AT(m.b2, i, j) -= lr* MAT_AT(g.b2, i, j);
        }
    }
}

float td[]= {
    0, 0, 0, 
    0, 1, 1,
    1, 0, 1,
    1, 1, 0
};

int main(void){
    // Implementing XOR model with tinyml framework
    srand((unsigned int) time(NULL));
    size_t stride= 3;
    size_t n= sizeof(td)/ sizeof(td[0])/stride;
    //printf("n: %d\n", n);

    Mat ti= (Mat){
        .rows= n,
        .cols= 2,
        .stride= stride,
        .es= td
    };

    Mat to= (Mat){
        .rows= n,
        .cols= 1,
        .stride= stride,
        .es= td + 2
    };

    MAT_PRINT(ti);
    MAT_PRINT(to);

    MAT_PRINT(mat_row(ti, 0));

    float step= 1e-3, lr= 1e-1;
    size_t epochs= 100000;

    Xor m= xor_alloc();
    Xor g= xor_alloc();

    for(size_t epoch=0; epoch< epochs; ++epoch){
        printf("cost at epoch: %zu: %f\n",epoch, cost(m, ti, to));
        finite_diff(m, g, ti, to, step);
        update_weights(m, g, lr);
    }



    printf("\n--------------------------------------------------------\n");
    for(size_t i=0; i<2; ++i){
        for(size_t j=0; j< 2; ++j){
            MAT_AT(m.a0, 0, 0)= i;
            MAT_AT(m.a0, 0, 1)= j;
            forward(m);
            printf("%zu ^ %zu = %f\n", i, j, forward(m));
        }
    }

    return 0;
}