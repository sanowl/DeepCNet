#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>

#define EPSILON 1e-8
#define MAX_LINE_LENGTH 1000

typedef enum {
    SIGMOID,
    RELU,
    TANH,
    LEAKY_RELU,
    ELU,
    SOFTMAX
} ActivationFunction;

typedef enum {
    MSE,
    CROSS_ENTROPY,
    BINARY_CROSS_ENTROPY
} LossFunction;

typedef enum {
    SGD,
    ADAM,
    RMSPROP,
    ADAGRAD
} OptimizerType;

typedef struct {
    int num_layers;
    int* neurons_per_layer;
    double** activations;
    double*** weights;
    double** biases;
    ActivationFunction* activation_functions;
    double** batch_norm_gamma;
    double** batch_norm_beta;
    double** batch_norm_moving_mean;
    double** batch_norm_moving_var;
    double dropout_rate;
} NeuralNetwork;

typedef struct {
    OptimizerType type;
    double learning_rate;
    double beta1;
    double beta2;
    double*** m;
    double*** v;
    double** m_bias;
    double** v_bias;
    int t;
} Optimizer;

typedef struct {
    double l1_lambda;
    double l2_lambda;
} Regularization;

// Function prototypes
NeuralNetwork* create_neural_network(int num_layers, int* neurons_per_layer, ActivationFunction* activation_functions);
void free_neural_network(NeuralNetwork* nn);
Optimizer* create_optimizer(NeuralNetwork* nn, OptimizerType type, double learning_rate, double beta1, double beta2);
void free_optimizer(Optimizer* optimizer, NeuralNetwork* nn);
void forward_pass(NeuralNetwork* nn, double* input, int is_training);
void backward_pass(NeuralNetwork* nn, double* target, double** delta, LossFunction loss_func);
void update_weights(NeuralNetwork* nn, Optimizer* optimizer, double** delta, Regularization* reg);
double compute_loss(NeuralNetwork* nn, double* target, LossFunction loss_func);
void predict(NeuralNetwork* nn, double* input, double* output);
void evaluate(NeuralNetwork* nn, double** inputs, double** targets, int num_samples, LossFunction loss_func);
void train(NeuralNetwork* nn, Optimizer* optimizer, double** inputs, double** targets, int num_samples, int max_epochs, int batch_size, double early_stopping_threshold, LossFunction loss_func, Regularization* reg);
void save_network(NeuralNetwork* nn, const char* filename);
NeuralNetwork* load_network(const char* filename);

// Activation functions and their derivatives
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double sigmoid_derivative(double x) { double s = sigmoid(x); return s * (1 - s); }
double relu(double x) { return x > 0 ? x : 0; }
double relu_derivative(double x) { return x > 0 ? 1 : 0; }
double tanh_activation(double x) { return tanh(x); }
double tanh_derivative(double x) { double t = tanh(x); return 1 - t * t; }
double leaky_relu(double x) { return x > 0 ? x : 0.01 * x; }
double leaky_relu_derivative(double x) { return x > 0 ? 1 : 0.01; }
double elu(double x) { return x > 0 ? x : exp(x) - 1; }
double elu_derivative(double x) { return x > 0 ? 1 : exp(x); }

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
        case TANH: return tanh_activation(x);
        case LEAKY_RELU: return leaky_relu(x);
        case ELU: return elu(x);
        case SOFTMAX: return x; // Softmax is applied to the whole layer
        default: fprintf(stderr, "Unknown activation function\n"); exit(1);
    }
}

double apply_activation_derivative(ActivationFunction func, double x) {
    switch (func) {
        case SIGMOID: return sigmoid_derivative(x);
        case RELU: return relu_derivative(x);
        case TANH: return tanh_derivative(x);
        case LEAKY_RELU: return leaky_relu_derivative(x);
        case ELU: return elu_derivative(x);
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
    
    nn->batch_norm_gamma = malloc((num_layers - 1) * sizeof(double*));
    nn->batch_norm_beta = malloc((num_layers - 1) * sizeof(double*));
    nn->batch_norm_moving_mean = malloc((num_layers - 1) * sizeof(double*));
    nn->batch_norm_moving_var = malloc((num_layers - 1) * sizeof(double*));
    
    nn->dropout_rate = 0.5; // Default dropout rate
    
    for (int i = 0; i < num_layers; i++) {
        nn->activations[i] = malloc(neurons_per_layer[i] * sizeof(double));
        
        if (i < num_layers - 1) {
            nn->weights[i] = malloc(neurons_per_layer[i] * sizeof(double*));
            for (int j = 0; j < neurons_per_layer[i]; j++) {
                nn->weights[i][j] = malloc(neurons_per_layer[i+1] * sizeof(double));
                for (int k = 0; k < neurons_per_layer[i+1]; k++) {
                    nn->weights[i][j][k] = ((double)rand() / RAND_MAX) * sqrt(2.0 / neurons_per_layer[i]); // He initialization
                }
            }
            
            nn->biases[i] = calloc(neurons_per_layer[i+1], sizeof(double));
            
            nn->batch_norm_gamma[i] = malloc(neurons_per_layer[i+1] * sizeof(double));
            nn->batch_norm_beta[i] = malloc(neurons_per_layer[i+1] * sizeof(double));
            nn->batch_norm_moving_mean[i] = calloc(neurons_per_layer[i+1], sizeof(double));
            nn->batch_norm_moving_var[i] = malloc(neurons_per_layer[i+1] * sizeof(double));
            
            for (int j = 0; j < neurons_per_layer[i+1]; j++) {
                nn->batch_norm_gamma[i][j] = 1.0;
                nn->batch_norm_beta[i][j] = 0.0;
                nn->batch_norm_moving_var[i][j] = 1.0;
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
            free(nn->batch_norm_gamma[i]);
            free(nn->batch_norm_beta[i]);
            free(nn->batch_norm_moving_mean[i]);
            free(nn->batch_norm_moving_var[i]);
        }
    }
    free(nn->activations);
    free(nn->weights);
    free(nn->biases);
    free(nn->neurons_per_layer);
    free(nn->activation_functions);
    free(nn->batch_norm_gamma);
    free(nn->batch_norm_beta);
    free(nn->batch_norm_moving_mean);
    free(nn->batch_norm_moving_var);
    free(nn);
}

Optimizer* create_optimizer(NeuralNetwork* nn, OptimizerType type, double learning_rate, double beta1, double beta2) {
    Optimizer* optimizer = malloc(sizeof(Optimizer));
    optimizer->type = type;
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

void free_optimizer(Optimizer* optimizer, NeuralNetwork* nn) {
    for (int i = 0; i < nn->num_layers - 1; i++) {
        for (int j = 0; j < nn->neurons_per_layer[i]; j++) {
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

void batch_normalize(NeuralNetwork* nn, int layer, int is_training) {
    double epsilon = 1e-8;
    int n = nn->neurons_per_layer[layer];
    double* input = nn->activations[layer];
    double* gamma = nn->batch_norm_gamma[layer-1];
    double* beta = nn->batch_norm_beta[layer-1];
    double* moving_mean = nn->batch_norm_moving_mean[layer-1];
    double* moving_var = nn->batch_norm_moving_var[layer-1];

    if (is_training) {
        double mean = 0, var = 0;
        for (int i = 0; i < n; i++) {
            mean += input[i];
        }
        mean /= n;

        for (int i = 0; i < n; i++) {
            double diff = input[i] - mean;
            var += diff * diff;
        }
        var /= n;

        // Update moving average and variance
        double momentum = 0.9;
        for (int i = 0; i < n; i++) {
            moving_mean[i] = momentum * moving_mean[i] + (1 - momentum) * mean;
            moving_var[i] = momentum * moving_var[i] + (1 - momentum) * var;
        }

        // Normalize and scale
        for (int i = 0; i < n; i++) {
            input[i] = gamma[i] * (input[i] - mean) / sqrt(var + epsilon) + beta[i];
        }
    } else {
        // Use moving average and variance for inference
        for (int i = 0; i < n; i++) {
            input[i] = gamma[i] * (input[i] - moving_mean[i]) / sqrt(moving_var[i] + epsilon) + beta[i];
        }
    }
}

void forward_pass(NeuralNetwork* nn, double* input, int is_training) {
    memcpy(nn->activations[0], input, nn->neurons_per_layer[0] * sizeof(double));
    
    for (int l = 1; l < nn->num_layers; l++) {
        for (int j = 0; j < nn->neurons_per_layer[l]; j++) {
            double sum = nn->biases[l-1][j];
            for (int i = 0; i < nn->neurons_per_layer[l-1]; i++) {
                sum += nn->activations[l-1][i] * nn->weights[l-1][i][j];
            }
            nn->activations[l][j] = sum;
        }
        
        // Apply batch normalization
        batch_normalize(nn, l, is_training);
        
        // Apply activation function
        for (int j = 0; j < nn->neurons_per_layer[l]; j++) {
            nn->activations[l][j] = apply_activation(nn->activation_functions[l-1], nn->activations[l][j]);
        }
        
        // Apply dropout during training
        if (is_training && l < nn->num_layers - 1) {
            for (int j = 0; j < nn->neurons_per_layer[l]; j++) {
                if ((double)rand() / RAND_MAX < nn->dropout_rate) {
                    nn->activations[l][j] = 0;
                } else {
                    nn->activations[l][j] /= (1 - nn->dropout_rate);
                }
            }
        }
    }
}

void backward_pass(NeuralNetwork* nn, double* target, double** delta, LossFunction loss_func) {
    int output_layer = nn->num_layers - 1;
    
    // Compute delta for output layer
    for (int j = 0; j < nn->neurons_per_layer[output_layer]; j++) {
        double output = nn->activations[output_layer][j];
        double error;
        switch (loss_func) {
            case MSE:
                error = target[j] - output;
                break;
            case CROSS_ENTROPY:
            case BINARY_CROSS_ENTROPY:
                error = target[j] / output - (1 - target[j]) / (1 - output);
                break;
        }
        delta[output_layer][j] = error * apply_activation_derivative(nn->activation_functions[output_layer-1], output);
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

void update_weights(NeuralNetwork* nn, Optimizer* optimizer, double** delta, Regularization* reg) {
    optimizer->t++;
    double lr_t = optimizer->learning_rate;
    
    if (optimizer->type == ADAM || optimizer->type == RMSPROP) {
        lr_t *= sqrt(1 - pow(optimizer->beta2, optimizer->t)) / (1 - pow(optimizer->beta1, optimizer->t));
    }
    
    for (int l = 0; l < nn->num_layers - 1; l++) {
        for (int i = 0; i < nn->neurons_per_layer[l]; i++) {
            for (int j = 0; j < nn->neurons_per_layer[l+1]; j++) {
                double grad = delta[l+1][j] * nn->activations[l][i];
                
                // Apply regularization
                if (reg) {
                    grad += reg->l2_lambda * nn->weights[l][i][j];
                    grad += reg->l1_lambda * (nn->weights[l][i][j] > 0 ? 1 : -1);
                }
                
               switch (optimizer->type) {
                    case SGD:
                        nn->weights[l][i][j] += lr_t * grad;
                        break;
                    case ADAM:
                    case RMSPROP:
                        optimizer->m[l][i][j] = optimizer->beta1 * optimizer->m[l][i][j] + (1 - optimizer->beta1) * grad;
                        optimizer->v[l][i][j] = optimizer->beta2 * optimizer->v[l][i][j] + (1 - optimizer->beta2) * grad * grad;
                        nn->weights[l][i][j] += lr_t * optimizer->m[l][i][j] / (sqrt(optimizer->v[l][i][j]) + EPSILON);
                        break;
                    case ADAGRAD:
                        optimizer->v[l][i][j] += grad * grad;
                        nn->weights[l][i][j] += lr_t * grad / (sqrt(optimizer->v[l][i][j]) + EPSILON);
                        break;
                }
            }
        }
        
        for (int j = 0; j < nn->neurons_per_layer[l+1]; j++) {
            double grad = delta[l+1][j];
            
            switch (optimizer->type) {
                case SGD:
                    nn->biases[l][j] += lr_t * grad;
                    break;
                case ADAM:
                case RMSPROP:
                    optimizer->m_bias[l][j] = optimizer->beta1 * optimizer->m_bias[l][j] + (1 - optimizer->beta1) * grad;
                    optimizer->v_bias[l][j] = optimizer->beta2 * optimizer->v_bias[l][j] + (1 - optimizer->beta2) * grad * grad;
                    nn->biases[l][j] += lr_t * optimizer->m_bias[l][j] / (sqrt(optimizer->v_bias[l][j]) + EPSILON);
                    break;
                case ADAGRAD:
                    optimizer->v_bias[l][j] += grad * grad;
                    nn->biases[l][j] += lr_t * grad / (sqrt(optimizer->v_bias[l][j]) + EPSILON);
                    break;
            }
        }
    }
}

double compute_loss(NeuralNetwork* nn, double* target, LossFunction loss_func) {
    int output_layer = nn->num_layers - 1;
    double loss = 0.0;
    
    switch (loss_func) {
        case MSE:
            for (int i = 0; i < nn->neurons_per_layer[output_layer]; i++) {
                double error = target[i] - nn->activations[output_layer][i];
                loss += error * error;
            }
            loss /= (2 * nn->neurons_per_layer[output_layer]);
            break;
        case CROSS_ENTROPY:
        case BINARY_CROSS_ENTROPY:
            for (int i = 0; i < nn->neurons_per_layer[output_layer]; i++) {
                loss -= target[i] * log(nn->activations[output_layer][i] + EPSILON) +
                        (1 - target[i]) * log(1 - nn->activations[output_layer][i] + EPSILON);
            }
            loss /= nn->neurons_per_layer[output_layer];
            break;
    }
    
    return loss;
}

void train(NeuralNetwork* nn, Optimizer* optimizer, double** inputs, double** targets, int num_samples, int max_epochs, int batch_size, double early_stopping_threshold, LossFunction loss_func, Regularization* reg) {
    double** delta = malloc(nn->num_layers * sizeof(double*));
    for (int i = 0; i < nn->num_layers; i++) {
        delta[i] = malloc(nn->neurons_per_layer[i] * sizeof(double));
    }
    
    double best_loss = DBL_MAX;
    int patience = 10;
    int patience_counter = 0;
    
    for (int epoch = 0; epoch < max_epochs; epoch++) {
        double total_loss = 0.0;
        
        // Shuffle the training data
        for (int i = 0; i < num_samples; i++) {
            int j = rand() % num_samples;
            double* temp_input = inputs[i];
            inputs[i] = inputs[j];
            inputs[j] = temp_input;
            double* temp_target = targets[i];
            targets[i] = targets[j];
            targets[j] = temp_target;
        }
        
        for (int batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
            int batch_end = batch_start + batch_size;
            if (batch_end > num_samples) batch_end = num_samples;
            
            for (int sample = batch_start; sample < batch_end; sample++) {
                forward_pass(nn, inputs[sample], 1);
                backward_pass(nn, targets[sample], delta, loss_func);
                update_weights(nn, optimizer, delta, reg);
                
                total_loss += compute_loss(nn, targets[sample], loss_func);
            }
        }
        
        double avg_loss = total_loss / num_samples;
        
        if (epoch % 100 == 0) {
            printf("Epoch %d, Average Loss: %f\n", epoch + 1, avg_loss);
        }
        
        if (avg_loss < best_loss - early_stopping_threshold) {
            best_loss = avg_loss;
            patience_counter = 0;
        } else {
            patience_counter++;
            if (patience_counter >= patience) {
                printf("Early stopping at epoch %d\n", epoch + 1);
                break;
            }
        }
    }
    
    for (int i = 0; i < nn->num_layers; i++) {
        free(delta[i]);
    }
    free(delta);
}

void predict(NeuralNetwork* nn, double* input, double* output) {
    forward_pass(nn, input, 0);
    int output_layer = nn->num_layers - 1;
    memcpy(output, nn->activations[output_layer], nn->neurons_per_layer[output_layer] * sizeof(double));
}

void evaluate(NeuralNetwork* nn, double** inputs, double** targets, int num_samples, LossFunction loss_func) {
    double total_loss = 0.0;
    int correct_predictions = 0;
    int output_size = nn->neurons_per_layer[nn->num_layers - 1];
    double* prediction = malloc(output_size * sizeof(double));

    for (int i = 0; i < num_samples; i++) {
        predict(nn, inputs[i], prediction);
        total_loss += compute_loss(nn, targets[i], loss_func);

        if (output_size == 1) {
            // Binary classification
            if ((prediction[0] >= 0.5 && targets[i][0] == 1) || (prediction[0] < 0.5 && targets[i][0] == 0)) {
                correct_predictions++;
            }
        } else {
            // Multi-class classification
            int predicted_class = 0;
            int target_class = 0;
            for (int j = 1; j < output_size; j++) {
                if (prediction[j] > prediction[predicted_class]) predicted_class = j;
                if (targets[i][j] > targets[i][target_class]) target_class = j;
            }
            if (predicted_class == target_class) correct_predictions++;
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
    srand(time(NULL));

    // Define the neural network architecture
    int num_layers = 4;
    int neurons_per_layer[] = {2, 5, 3, 1};
    ActivationFunction activation_functions[] = {RELU, RELU, SIGMOID};

    // Create the neural network
    NeuralNetwork* nn = create_neural_network(num_layers, neurons_per_layer, activation_functions);

    // Create the optimizer
    Optimizer* optimizer = create_optimizer(nn, ADAM, 0.001, 0.9, 0.999);

    // Define regularization
    Regularization reg = {.l1_lambda = 0.0001, .l2_lambda = 0.0001};

    // Generate some example data (XOR problem)
    int num_samples = 4;
    double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double targets[4][1] = {{0}, {1}, {1}, {0}};

    double** input_ptr = malloc(num_samples * sizeof(double*));
    double** target_ptr = malloc(num_samples * sizeof(double*));
    for (int i = 0; i < num_samples; i++) {
        input_ptr[i] = inputs[i];
        target_ptr[i] = targets[i];
    }

    // Train the network
    printf("Training the neural network...\n");
    train(nn, optimizer, input_ptr, target_ptr, num_samples, 10000, 4, 1e-6, BINARY_CROSS_ENTROPY, &reg);

    // Evaluate the trained network
    printf("\nEvaluating the trained network:\n");
    evaluate(nn, input_ptr, target_ptr, num_samples, BINARY_CROSS_ENTROPY);

    // Make predictions
    printf("\nMaking predictions:\n");
    double prediction[1];
    for (int i = 0; i < num_samples; i++) {
        predict(nn, inputs[i], prediction);
        printf("Input: [%.0f, %.0f], Predicted Output: %.4f, Actual Output: %.0f\n",
               inputs[i][0], inputs[i][1], prediction[0], targets[i][0]);
    }

    // Clean up
    free_neural_network(nn);
    free_optimizer(optimizer, nn);
    free(input_ptr);
    free(target_ptr);

    return 0;
}

