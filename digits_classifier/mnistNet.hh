#include <vector>
#include <iostream>
#include <cstdio>
#include <ctime>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/framework/gradients.h"

using namespace tensorflow;
using namespace tensorflow::ops;

class MnistNet{
private:
    Scope scope;
    Placeholder *x, *y;
    // weights and biases
    Variable *w1, *w2, *b1, *b2;
    Assign *assign_w1, *assign_w2, *assign_b1, *assign_b2;
    // activation for layers
    Sigmoid *layer_1, *layer_2;
    // loss
    ReduceMean *loss;
    // apply gradient descend
    ApplyGradientDescent *apply_w1, *apply_w2, *apply_b1, *apply_b2;
    // create session to run the graph
    ClientSession *session;
    // hyperparameters
    const int size_input       = 784;
    const int size_first_layer = 30;
    const int size_output      = 10;

    const int minibatch_size = 50;
    const int default_number_of_epochs = 50;
    const float default_learning_rate = 1.0;

    // other constants
    const int image_size = 784;
    const int label_size = 10;

    std::vector<float> get_flat_images(std::vector< std::vector<int> >images);
    std::vector<float> get_flat_one_hot_labels(std::vector<int> labels);

public:
    // Building the computational graph for the net
    MnistNet();

    // Traing the network with the input data and passed number_of_epochs
    // Pass NULL to use default_number_of_epochs  
    // TODO use overloading
    void train(std::vector< std::vector<int> >images,
               std::vector<int> labels,
               int number_of_epochs);

    // returns a vector with the prediction for each image
    std::vector<int> predict(std::vector< std::vector<int> >images);
};