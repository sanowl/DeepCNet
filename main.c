#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>  // Include this for time functions

#define EPSILON 1e-8

typedef enum {
    SIGMOID,
    RELU,
    SOFTMAX
} ActivationFunction;

typedef struct {
    int num_layers;
    int* neurons_per_layer;
    double** activations;
    double*** weights;
    double** biases;
    ActivationFunction* activation_functions;
} NeuralNetwork;

typedef struct {
    double learning_rate;
    double beta1;
    double beta2;
    double*** m;
    double*** v;
    double** m_bias;
    double** v_bias;
    int t;
} AdamOptimizer;

// Function prototypes
NeuralNetwork* create_neural_network(int num_layers, int* neurons_per_layer, ActivationFunction* activation_functions);
void free_neural_network(NeuralNetwork* nn);
AdamOptimizer* create_adam_optimizer(NeuralNetwork* nn, double learning_rate, double beta1, double beta2);
void free_adam_optimizer(AdamOptimizer* optimizer, int num_layers);
void forward_pass(NeuralNetwork* nn, double* input);
void backward_pass(NeuralNetwork* nn, double* target, double** delta);
void update_weights_adam(NeuralNetwork* nn, AdamOptimizer* optimizer, double** delta);
double compute_loss(NeuralNetwork* nn, double* target);
void predict(NeuralNetwork* nn, double* input, double* output);
void evaluate(NeuralNetwork* nn, double** inputs, double** targets, int num_samples);

// Activation functions and their derivatives
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

double relu(double x) {
    return x > 0 ? x : 0;
}

double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

void softmax(double* input, int size) {
    double max = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max) max = input[i];
    }

    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        input[i] = exp(input[i] - max);
        sum += input[i];
    }

    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

double apply_activation(ActivationFunction func, double x) {
    switch (func) {
        case SIGMOID: return sigmoid(x);
        case RELU: return relu(x);
        case SOFTMAX: return x; // Softmax is applied to the whole layer, not individual neurons
        default: fprintf(stderr, "Unknown activation function\n"); exit(1);
    }
}

double apply_activation_derivative(ActivationFunction func, double x) {
    switch (func) {
        case SIGMOID: return sigmoid_derivative(x);
        case RELU: return relu_derivative(x);
        case SOFTMAX: return 1; // For softmax, this is handled differently in backward pass
        default: fprintf(stderr, "Unknown activation function\n"); exit(1);
    }
}

NeuralNetwork* create_neural_network(int num_layers, int* neurons_per_layer, ActivationFunction* activation_functions) {
    NeuralNetwork* nn = malloc(sizeof(NeuralNetwork));
    nn->num_layers = num_layers;
    nn->neurons_per_layer = malloc(num_layers * sizeof(int));
    memcpy(nn->neurons_per_layer, neurons_per_layer, num_layers * sizeof(int));

    nn->activation_functions = malloc((num_layers - 1) * sizeof(ActivationFunction));
    memcpy(nn->activation_functions, activation_functions, (num_layers - 1) * sizeof(ActivationFunction));

    nn->activations = malloc(num_layers * sizeof(double*));
    nn->weights = malloc((num_layers - 1) * sizeof(double**));
    nn->biases = malloc((num_layers - 1) * sizeof(double*));

    for (int i = 0; i < num_layers; i++) {
        nn->activations[i] = malloc(neurons_per_layer[i] * sizeof(double));

        if (i < num_layers - 1) {
            nn->weights[i] = malloc(neurons_per_layer[i] * sizeof(double*));
            for (int j = 0; j < neurons_per_layer[i]; j++) {
                nn->weights[i][j] = malloc(neurons_per_layer[i+1] * sizeof(double));
                for (int k = 0; k < neurons_per_layer[i+1]; k++) {
                    nn->weights[i][j][k] = ((double)rand() / RAND_MAX) * 2 - 1; // Initialize between -1 and 1
                }
            }

            nn->biases[i] = malloc(neurons_per_layer[i+1] * sizeof(double));
            for (int j = 0; j < neurons_per_layer[i+1]; j++) {
                nn->biases[i][j] = 0; // Initialize biases to 0
            }
        }
    }

    return nn;
}

void free_neural_network(NeuralNetwork* nn) {
    for (int i = 0; i < nn->num_layers; i++) {
        free(nn->activations[i]);
        if (i < nn->num_layers - 1) {
            for (int j = 0; j < nn->neurons_per_layer[i]; j++) {
                free(nn->weights[i][j]);
            }
            free(nn->weights[i]);
            free(nn->biases[i]);
        }
    }
    free(nn->activations);
    free(nn->weights);
    free(nn->biases);
    free(nn->neurons_per_layer);
    free(nn->activation_functions);
    free(nn);
}

AdamOptimizer* create_adam_optimizer(NeuralNetwork* nn, double learning_rate, double beta1, double beta2) {
    AdamOptimizer* optimizer = malloc(sizeof(AdamOptimizer));
    optimizer->learning_rate = learning_rate;
    optimizer->beta1 = beta1;
    optimizer->beta2 = beta2;
    optimizer->t = 0;

    optimizer->m = malloc((nn->num_layers - 1) * sizeof(double**));
    optimizer->v = malloc((nn->num_layers - 1) * sizeof(double**));
    optimizer->m_bias = malloc((nn->num_layers - 1) * sizeof(double*));
    optimizer->v_bias = malloc((nn->num_layers - 1) * sizeof(double*));

    for (int i = 0; i < nn->num_layers - 1; i++) {
        optimizer->m[i] = malloc(nn->neurons_per_layer[i] * sizeof(double*));
        optimizer->v[i] = malloc(nn->neurons_per_layer[i] * sizeof(double*));
        for (int j = 0; j < nn->neurons_per_layer[i]; j++) {
            optimizer->m[i][j] = calloc(nn->neurons_per_layer[i+1], sizeof(double));
            optimizer->v[i][j] = calloc(nn->neurons_per_layer[i+1], sizeof(double));
        }
        optimizer->m_bias[i] = calloc(nn->neurons_per_layer[i+1], sizeof(double));
        optimizer->v_bias[i] = calloc(nn->neurons_per_layer[i+1], sizeof(double));
    }

    return optimizer;
}

void free_adam_optimizer(AdamOptimizer* optimizer, int num_layers) {
    for (int i = 0; i < num_layers - 1; i++) {
        for (int j = 0; j < optimizer->m[i][0]; j++) {
            free(optimizer->m[i][j]);
            free(optimizer->v[i][j]);
        }
        free(optimizer->m[i]);
        free(optimizer->v[i]);
        free(optimizer->m_bias[i]);
        free(optimizer->v_bias[i]);
    }
    free(optimizer->m);
    free(optimizer->v);
    free(optimizer->m_bias);
    free(optimizer->v_bias);
    free(optimizer);
}

void forward_pass(NeuralNetwork* nn, double* input) {
    memcpy(nn->activations[0], input, nn->neurons_per_layer[0] * sizeof(double));

    for (int l = 1; l < nn->num_layers; l++) {
        for (int j = 0; j < nn->neurons_per_layer[l]; j++) {
            double sum = nn->biases[l-1][j];
            for (int i = 0; i < nn->neurons_per_layer[l-1]; i++) {
                sum += nn->activations[l-1][i] * nn->weights[l-1][i][j];
            }
            nn->activations[l][j] = apply_activation(nn->activation_functions[l-1], sum);
        }

        if (nn->activation_functions[l-1] == SOFTMAX) {
            softmax(nn->activations[l], nn->neurons_per_layer[l]);
        }
    }
}

void backward_pass(NeuralNetwork* nn, double* target, double** delta) {
    int output_layer = nn->num_layers - 1;

    // Compute delta for output layer
    for (int j = 0; j < nn->neurons_per_layer[output_layer]; j++) {
        double output = nn->activations[output_layer][j];
        double error = target[j] - output;
        if (nn->activation_functions[output_layer-1] == SOFTMAX) {
            delta[output_layer][j] = error;
        } else {
            delta[output_layer][j] = error * apply_activation_derivative(nn->activation_functions[output_layer-1], output);
        }
    }

    // Backpropagate the error
    for (int l = output_layer - 1; l > 0; l--) {
        for (int i = 0; i < nn->neurons_per_layer[l]; i++) {
            double error = 0.0;
            for (int j = 0; j < nn->neurons_per_layer[l+1]; j++) {
                error += delta[l+1][j] * nn->weights[l][i][j];
            }
            delta[l][i] = error * apply_activation_derivative(nn->activation_functions[l-1], nn->activations[l][i]);
        }
    }
}

void update_weights_adam(NeuralNetwork* nn, AdamOptimizer* optimizer, double** delta) {
    optimizer->t++;
    double lr_t = optimizer->learning_rate * sqrt(1 - pow(optimizer->beta2, optimizer->t)) / (1 - pow(optimizer->beta1, optimizer->t));

    for (int l = 0; l < nn->num_layers - 1; l++) {
        for (int i = 0; i < nn->neurons_per_layer[l]; i++) {
            for (int j = 0; j < nn->neurons_per_layer[l+1]; j++) {
                double grad = delta[l+1][j] * nn->activations[l][i];
                optimizer->m[l][i][j] = optimizer->beta1 * optimizer->m[l][i][j] + (1 - optimizer->beta1) * grad;
                optimizer->v[l][i][j] = optimizer->beta2 * optimizer->v[l][i][j] + (1 - optimizer->beta2) * grad * grad;
                nn->weights[l][i][j] += lr_t * optimizer->m[l][i][j] / (sqrt(optimizer->v[l][i][j]) + EPSILON);
            }
        }

        for (int j = 0; j < nn->neurons_per_layer[l+1]; j++) {
            double grad = delta[l+1][j];
            optimizer->m_bias[l][j] = optimizer->beta1 * optimizer->m_bias[l][j] + (1 - optimizer->beta1) * grad;
            optimizer->v_bias[l][j] = optimizer->beta2 * optimizer->v_bias[l][j] + (1 - optimizer->beta2) * grad * grad;
            nn->biases[l][j] += lr_t * optimizer->m_bias[l][j] / (sqrt(optimizer->v_bias[l][j]) + EPSILON);
        }
    }
}

double compute_loss(NeuralNetwork* nn, double* target) {
    int output_layer = nn->num_layers - 1;
    double loss = 0.0;

    for (int i = 0; i < nn->neurons_per_layer[output_layer]; i++) {
        double error = target[i] - nn->activations[output_layer][i];
        loss += error * error;
    }

    return loss / (2 * nn->neurons_per_layer[output_layer]);
}

void train(NeuralNetwork* nn, AdamOptimizer* optimizer, double** inputs, double** targets, int num_samples, int epochs) {
    double** delta = malloc(nn->num_layers * sizeof(double*));
    for (int i = 0; i < nn->num_layers; i++) {
        delta[i] = malloc(nn->neurons_per_layer[i] * sizeof(double));
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;

        for (int sample = 0; sample < num_samples; sample++) {
            forward_pass(nn, inputs[sample]);
            backward_pass(nn, targets[sample], delta);
            update_weights_adam(nn, optimizer, delta);

            total_loss += compute_loss(nn, targets[sample]);
        }

        if (epoch % 1000 == 0) {
            printf("Epoch %d, Average Loss: %f\n", epoch + 1, total_loss / num_samples);
        }
    }

    for (int i = 0; i < nn->num_layers; i++) {
        free(delta[i]);
    }
    free(delta);
}

void predict(NeuralNetwork* nn, double* input, double* output) {
    forward_pass(nn, input);
    int output_layer = nn->num_layers - 1;
    memcpy(output, nn->activations[output_layer], nn->neurons_per_layer[output_layer] * sizeof(double));
}

void evaluate(NeuralNetwork* nn, double** inputs, double** targets, int num_samples) {
    double total_loss = 0.0;
    int correct_predictions = 0;
    int output_size = nn->neurons_per_layer[nn->num_layers - 1];
    double* prediction = malloc(output_size * sizeof(double));

    for (int i = 0; i < num_samples; i++) {
        predict(nn, inputs[i], prediction);
        total_loss += compute_loss(nn, targets[i]);

        // For binary classification
        if (output_size == 1) {
            if ((prediction[0] >= 0.5 && targets[i][0] == 1) || (prediction[0] < 0.5 && targets[i][0] == 0)) {
                correct_predictions++;
            }
        } 
        // For multi-class classification
        else {
            int predicted_class = 0;
            int target_class = 0;
            for (int j = 1; j < output_size; j++) {
                if (prediction[j] > prediction[predicted_class]) {
                    predicted_class = j;
                }
                if (targets[i][j] > targets[i][target_class]) {
                    target_class = j;
                }
            }
            if (predicted_class == target_class) {
                correct_predictions++;
            }
        }
    }

    double accuracy = (double)correct_predictions / num_samples;
    double avg_loss = total_loss / num_samples;

    printf("Evaluation results:\n");
    printf("Average loss: %f\n", avg_loss);
    printf("Accuracy: %f%%\n", accuracy * 100);

    free(prediction);
}

int main() {
    srand(time(NULL));  // Initialize random seed

    int num_layers = 3;
    int neurons_per_layer[] = {2, 4, 1};
    ActivationFunction activation_functions[] = {SIGMOID, SIGMOID};
    
    NeuralNetwork* nn = create_neural_network(num_layers, neurons_per_layer, activation_functions);
    AdamOptimizer* optimizer = create_adam_optimizer(nn, 0.01, 0.9, 0.999);
    
    // XOR problem dataset
    double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double targets[4][1] = {{0}, {1}, {1}, {0}};
    
    double** input_ptr = malloc(4 * sizeof(double*));
    double** target_ptr = malloc(4 * sizeof(double*));
    
    for (int i = 0; i < 4; i++) {
        input_ptr[i] = inputs[i];
        target_ptr[i] = targets[i];
    }

    // Training
    printf("Training the neural network...\n");
    train(nn, optimizer, input_ptr, target_ptr, 4, 10000);

    // Evaluation
    printf("\nEvaluating the trained network:\n");
    evaluate(nn, input_ptr, target_ptr, 4);

    // Predictions
    printf("\nMaking predictions:\n");
    double prediction[1];
    for (int i = 0; i < 4; i++) {
        predict(nn, inputs[i], prediction);
        printf("Input: [%.0f, %.0f], Predicted Output: %.4f, Actual Output: %.0f\n", 
               inputs[i][0], inputs[i][1], prediction[0], targets[i][0]);
    }

    // Clean up
    free_neural_network(nn);
    free_adam_optimizer(optimizer, num_layers);
    free(input_ptr);
    free(target_ptr);

    return 0;
}

