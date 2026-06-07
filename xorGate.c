#include<stdio.h>
#include<time.h>
#include<math.h>
#include<string.h>
#include<stdlib.h>

float activation_fn(float y){ // Change this function to use other activation functions
    return 1/(1 + exp(-1 * y)); // sigmoid activation function
}

float neuron(float activations[], float weights[], float b, int n){
    float y= b;
    for(int i=0; i<n; i++){
        y += activations[i] * weights[i];
    }
    float z = activation_fn(y);
    return z;
}

float MSE(int dataset[][3], float w1[2], float w2[2], float w3[2], float b1, float b2, float b3, int n){
   float loss = 0.0f;
   float n1, n2, n3, diff;

   for(int i=0; i< n; i++){
    float activation[2] = {dataset[i][0], dataset[i][1]};
    n1 = neuron(activation, w1, b1, 2);
    n2 = neuron(activation, w2, b2, 2);
    float last_neuron_activation[2] = {n1, n2};
    n3 = neuron(last_neuron_activation, w3, b3, 2);

    diff = n3 - dataset[i][2];
    loss += diff * diff;
   }

   return loss/ n;
}

void train(int dataset[][3], float w1[2], float w2[2], float w3[2], float *b1, float *b2, float *b3, int n, float step, float lr, int epochs){

    //Modify weights of last neuron

    for(int i=0; i<= epochs; i++){
        float loss = MSE(dataset, w1, w2, w3, *b1, *b2, *b3, n);
        printf("loss after %d epochs: %f\n", i, loss);

        float step_changed_weights[2];
        float step_changed_loss;
        float step_changed_b;

        // Modifying first weight of last neuron
        memcpy(step_changed_weights, w3, sizeof(step_changed_weights));
        step_changed_weights[0] += step;
        step_changed_loss = MSE(dataset, w1, w2, step_changed_weights, *b1, *b2, *b3, n);
        float dLdw20 = (step_changed_loss - loss)/step;

        // Modifying second weight of last neuron
        memcpy(step_changed_weights, w3, sizeof(step_changed_weights));
        step_changed_weights[1] += step;
        step_changed_loss = MSE(dataset, w1, w2, step_changed_weights, *b1, *b2, *b3, n);
        float dLdw21 = (step_changed_loss - loss)/step;
        
        // Modifying first weight of second neuron
        memcpy(step_changed_weights, w2, sizeof(step_changed_weights));
        step_changed_weights[0] += step;
        step_changed_loss = MSE(dataset, w1, step_changed_weights, w3, *b1, *b2, *b3, n);
        float dLdw10 = (step_changed_loss - loss)/step;


        // Modifying second weight of second neuron
        memcpy(step_changed_weights, w2, sizeof(step_changed_weights));
        step_changed_weights[1] += step;
        step_changed_loss = MSE(dataset, w1, step_changed_weights, w3, *b1, *b2, *b3, n);
        float dLdw11 = (step_changed_loss - loss)/step;

        // Modifying first weight of first neuron
        memcpy(step_changed_weights, w1, sizeof(step_changed_weights));
        step_changed_weights[0] += step;
        step_changed_loss = MSE(dataset, step_changed_weights, w2, w3, *b1, *b2, *b3, n);
        float dLdw00 = (step_changed_loss - loss)/step;

        // Modifying second weight of first neuron
        memcpy(step_changed_weights, w1, sizeof(step_changed_weights));
        step_changed_weights[1] += step;
        step_changed_loss = MSE(dataset, step_changed_weights, w2, w3, *b1, *b2, *b3, n);
        float dLdw01 = (step_changed_loss - loss)/step;

        //Modifying the bias of last neuron
        step_changed_b = *b3;
        step_changed_b += step;

        step_changed_loss = MSE(dataset, w1, w2, w3, *b1, *b2, step_changed_b, n);
        float dLdb3= (step_changed_loss - loss)/step;

        //Modifying the bias of second neuron
        step_changed_b = *b2;
        step_changed_b += step;

        step_changed_loss = MSE(dataset, w1, w2, w3, *b1, step_changed_b, *b3, n);
        float dLdb2= (step_changed_loss - loss)/step;

        //Modifying the bias of first neuron
        step_changed_b = *b1;
        step_changed_b += step;

        step_changed_loss = MSE(dataset, w1, w2, w3, step_changed_b, *b2, *b3, n);
        float dLdb1= (step_changed_loss - loss)/step;


        w3[0] -= lr * dLdw20;
        w3[1] -= lr * dLdw21;
        w2[0] -= lr * dLdw10;
        w2[1] -= lr * dLdw11;
        w1[0] -= lr * dLdw00;
        w1[1] -= lr * dLdw01;
        *b3 -= lr * dLdb3;
        *b2 -= lr * dLdb2;
        *b1 -= lr * dLdb1;
    }
}

void test(int dataset[][3], float w1[2], float w2[2], float w3[2], float b1, float b2, float b3, int n){
   float loss = 0.0f;
   float n1, n2, n3;

   for(int i=0; i< n; i++){
    float activation[2] = {dataset[i][0], dataset[i][1]};
    n1 = neuron(activation, w1, b1, 2);
    n2 = neuron(activation, w2, b2, 2);
    float last_neuron_activation[2] = {n1, n2};
    n3 = neuron(last_neuron_activation, w3, b3, 2);

    printf("i1: %d\t, i2: %d\t, output: %f\n", (int) activation[0], (int) activation[1], n3);
   }

}

int main(){
    int training[][3] = {
        {0, 0, 0},
        {1, 0, 1},
        {0, 1, 1},
        {1, 1, 0}
    };
    int n= sizeof(training)/ sizeof(training[0]);

   float n1, n2, n3;

   srand((unsigned int) time(NULL));
   
   float w1[2] = {(float)rand() / RAND_MAX, (float)rand() / RAND_MAX};
   float w2[2] = {(float)rand() / RAND_MAX, (float)rand() / RAND_MAX};
   float w3[2] = {(float)rand() / RAND_MAX, (float)rand() / RAND_MAX};

   float b1 = (float)rand() / RAND_MAX;
   float b2 = (float)rand() / RAND_MAX;
   float b3 = (float)rand() / RAND_MAX;

    float loss = MSE(training, w1, w2, w3, b1, b2, b3, n);
    printf("loss: %f\n", loss);
   
    float step = 1e-3;
    float lr = 1e-1;
    int epochs= 1000000;
    train(training, w1, w2, w3, &b1, &b2, &b3, n, step, lr, epochs);

    test(training, w1, w2, w3, b1, b2, b3, n);



    return 0;
}
