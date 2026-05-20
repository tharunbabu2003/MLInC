/*
* Can be used for both AND, OR gates by just changing the training data
*/


#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<string.h>
#include<math.h>


float MSLoss(int dataset[][3], int n, float weights[2], float b){
    float cost = 0.0;
    for(int i = 0; i < n; i++){
        float y = 0.0;
        for(int j = 0; j < 2; j++){
            y += weights[j] * dataset[i][j];
        }
        y += b;
        float sigmoid_y = 1/ (1 + exp(-1 * y));
        /*
        * Sigmoid activation function is used to keep the output of the neuron within 0-1. This is best suited for logic gates because the possible outputs of logic gates are 0 and 1
        */
        float diff = sigmoid_y - dataset[i][2];
        cost += diff * diff;
    }
    return cost/n;
}

void train(int dataset[][3], float lr, float step, int epochs, int n, float w[2], float *b){
    //printf("in training...\n");
    float db = 0.0f;
    
    for(int epoch= 1; epoch<= epochs; epoch++){
        /*
        * Loss should calculated in each epoch to correctly calcalute the finite derivative because the weights would have changed during the previous epoch. The loss must be newly calculated to find out the finite derivative correctly
        * Loss with the latest weights are required to calculate finite derivative
        * finite derivative = (f(x + h) - f(x))/ h
        * Here f(x) is the loss of neural network with the lastest weights
        */
        float loss = MSLoss(dataset, n, w, *b);  
        float dw[2] = {0.0f, 0.0f};
        
        // dL/ dw0, dL/dw1
        for(int i= 0; i< 2;i++){
            float step_changed_weights[2];
            memcpy(step_changed_weights, w, sizeof(step_changed_weights));
            step_changed_weights[i] += step;
            float step_changed_loss= MSLoss(dataset, n, step_changed_weights, *b);
            dw[i]= (step_changed_loss - loss)/ step;
        }
        
        // w0 = w0 - (lr * dL/ dw0), w1 = w1 - (lr * dL/ dw1)
        for(int i=0; i<2; i++){
            w[i] -= lr*dw[i];
        }
        
        // b = b - (lr * dL/ db)
        db = (MSLoss(dataset, n, w, *b + step) - loss)/ step;
        *b -= lr*db;

        printf("loss after %d epochs= %f\n", epoch, MSLoss(dataset, n, w, *b));
    }
}

void testing(int dataset[][3], float weights[2], float b, int n){
    for(int i=0; i< n; i++){
        float z = weights[0] * dataset[i][0] + weights[1] * dataset[i][1] + b;
        float y = 1/ (1 + exp(-1 * z));
        printf("i1= %d\t, i2= %d\t, y= %f\n", dataset[i][0], dataset[i][1], y);
    }
}

int main(){
    int training[][3] = {
        {0, 0, 0},
        {0, 1, 0},
        {1, 0, 0},
        {1, 1, 1}
    };

    int n = sizeof(training) / sizeof(training[0]);

    srand((unsigned int) time(NULL));

    float weights[2] = {(float)rand() / RAND_MAX, (float)rand() / RAND_MAX};
    float b = (float) rand()/ RAND_MAX;
    printf("w0= %f, w1=%f, b= %f\n", weights[0], weights[1], b);

    float loss= MSLoss(training, n, weights, b);
    printf("loss: %f\n", loss);

    float step= 1e-1;
    float lr= 1e-1;
    int epochs= 150000;

    train(training, lr, step, epochs, n, weights, &b);
    
    printf("w0= %f, w1=%f, b= %f\n", weights[0], weights[1], b);

    testing(training, weights, b, n);

    return 0;
}