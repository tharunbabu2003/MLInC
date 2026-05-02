#include<stdio.h>
#include<stdlib.h>
#include<time.h>


float MSLoss(int dataset[][2], int n, float w, float b){
    float cost = 0.0;
    for(int i = 0; i < n; i++){
        float y = w * dataset[i][0] + b;
        //printf("model output= %f, actual output= %f\n", y, (float)dataset[i][1]);
        float diff = y - dataset[i][1];
        cost += diff * diff;
    }
    return cost/n;
}

void train(int dataset[][2], float lr, float step, int epochs, int n, float *w, float *b){
    float dw = 0.0f;
    float db = 0.0f;
    for(int i= 1; i<= epochs; i++){
        //printf("epoch= %d\n", i);

        // finite derivative - approximation of dL/dw, db/dw
        dw= (MSLoss(dataset, n, *w + step, *b) - MSLoss(dataset, n, *w, *b))/ step;
        db = (MSLoss(dataset, n, *w, *b + step) - MSLoss(dataset, n, *w, *b))/ step;

        *w -= lr*dw;
        *b -= lr*db;
        printf("loss after %d epochs= %f\n", i, MSLoss(dataset, n, *w, *b));
    }
}

int main(){
    int training[][2] = {
        {0, 0},
        {1, 2},
        {2, 4},
        {3, 6},
        {4, 8},
    };

    int n = sizeof(training) / sizeof(training[0]);

    srand((unsigned int) time(NULL));
    float w = (float)rand() / RAND_MAX;
    float b = (float) rand()/ RAND_MAX;
    printf("w= %f, b= %f\n", w, b);

    float loss= MSLoss(training, n, w, b);
    printf("loss: %f\n", loss);

    float step= 1e-2;
    float lr= 1e-2;
    int epochs= 2500;

    train(training, lr, step, epochs, n, &w, &b);
    
    printf("w= %f, b= %f\n", w, b);

    return 0;
}