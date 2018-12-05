#include <vector>
#include <iostream>
#include <cstdio>
#include <ctime>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/cc/framework/gradients.h"

#include "tensorflow/cc/digits_classifier_prj/mnist-loader/mnist-loader.hh"

using namespace tensorflow;
using namespace tensorflow::ops;

int main(){
    std::srand(std::time(NULL));
    
    std::vector< std::vector<int> > images = mnist::get_train_images();
    std::vector<int> labels = mnist::get_train_labels();

    std::vector<float> flat_images;
    for(std::vector<int> image: images){
        std::vector<float> float_image;
        for(int pixel: image){
            float_image.push_back((float)pixel);
        }
        flat_images.insert(flat_images.end(), float_image.begin(), float_image.end());
    }
    
    std::vector<float> flat_one_hot_labels;
    for(int label: labels){
        std::vector<float> tmp(10, 0.0f);
        tmp[label] = 1.0f;
        flat_one_hot_labels.insert(flat_one_hot_labels.end(), tmp.begin(), tmp.end());
    }

    Scope scope = Scope::NewRootScope();

    auto x = Placeholder(scope, DT_FLOAT);
    auto y = Placeholder(scope, DT_FLOAT);

    // hyperparameters
    const int size_input = images[0].size();
    const int size_first_layer = 30;
    const int size_output = 10;
    const float learning_rate = 1.0;
    const int number_of_epochs = 70;
    const int minibatch_size = 50;

    // weights init
    auto w1 = Variable(scope, {size_input, size_first_layer}, DT_FLOAT);
    auto assign_w1 = Assign(scope, w1, RandomNormal(scope, {size_input, size_first_layer}, DT_FLOAT));

    auto w2 = Variable(scope, {size_first_layer, size_output}, DT_FLOAT);
    auto assign_w2 = Assign(scope, w2, RandomNormal(scope, {size_first_layer, size_output}, DT_FLOAT));

    // bias init
    auto b1 = Variable(scope, {1, size_first_layer}, DT_FLOAT);
    auto assign_b1 = Assign(scope, b1, RandomNormal(scope, {1, size_first_layer}, DT_FLOAT));

    auto b2 = Variable(scope, {1, size_output}, DT_FLOAT);
    auto assign_b2 = Assign(scope, b2, RandomNormal(scope, {1, size_output}, DT_FLOAT));

    // layers
    auto layer_1 = Sigmoid(scope, Add(scope, MatMul(scope, x, w1), b1));
    auto layer_2 = Sigmoid(scope, Add(scope, MatMul(scope, layer_1, w2), b2));

    auto loss = ReduceMean(scope, Square(scope, Subtract(scope, layer_2, y)), {0, 1});

    std::vector<Output> grad_outputs;
    TF_CHECK_OK(AddSymbolicGradients(scope, {loss}, {w1, w2, b1, b2}, &grad_outputs));

    // update the weights and bias using gradient descent
    auto apply_w1 = ApplyGradientDescent(scope, w1, Cast(scope, learning_rate,  DT_FLOAT), {grad_outputs[0]});
    auto apply_w2 = ApplyGradientDescent(scope, w2, Cast(scope, learning_rate,  DT_FLOAT), {grad_outputs[1]});
    auto apply_b1 = ApplyGradientDescent(scope, b1, Cast(scope, learning_rate,  DT_FLOAT), {grad_outputs[2]});
    auto apply_b2 = ApplyGradientDescent(scope, b2, Cast(scope, learning_rate,  DT_FLOAT), {grad_outputs[3]});

    ClientSession session(scope);
    std::vector<Tensor> outputs;

    // init the weights and biases by running the assigns nodes once
    TF_CHECK_OK(session.Run({assign_w1, assign_w2, assign_b1, assign_b2}, nullptr));

    // training
    for(int epoch = 0; epoch < number_of_epochs; epoch++){
        for(int minibatch = 0; minibatch < images.size() / minibatch_size; minibatch++){
            std::vector<float> minibatch_images;
            std::vector<float> minibatch_labels;
            minibatch_images.reserve(minibatch_size * images[0].size());
            minibatch_labels.reserve(minibatch_size * 10);
            // select a bunch of ranom samples for the minibatch
            for(int sample = 0; sample < minibatch_size; sample++){
                int idx = std::rand() % images.size();
                std::copy(begin(flat_images) + idx * images[0].size(),
                          begin(flat_images) + idx * images[0].size() + images[0].size(),
                          std::back_inserter(minibatch_images));
                std::copy(begin(flat_one_hot_labels) + idx * 10,
                          begin(flat_one_hot_labels) + idx * 10 + 10,
                          std::back_inserter(minibatch_labels));
            }
            // create the tensors for the mini_batch
            Tensor x_minibatch(DataTypeToEnum<float>::v(), 
                               TensorShape{static_cast<int>(minibatch_size), static_cast<int>(images[0].size())});
            std::copy_n(minibatch_images.begin(), minibatch_images.size(), x_minibatch.flat<float>().data());

            Tensor y_minibatch(DataTypeToEnum<float>::v(), 
                               TensorShape{static_cast<int>(minibatch_size), 10});
            std::copy_n(minibatch_labels.begin(), minibatch_labels.size(), y_minibatch.flat<float>().data());

            // calculate the loss for the minibatch
            TF_CHECK_OK(session.Run({{x, x_minibatch}, {y, y_minibatch}}, {loss}, &outputs));
            // backpropagation (apply gradients)
            TF_CHECK_OK(session.Run({{x, x_minibatch}, {y, y_minibatch}}, {apply_w1, apply_w2, apply_b1, apply_b2, layer_2}, nullptr));

            //std::cout << "  minibatch: " << minibatch << " - loss: " << outputs[0].scalar<float>() << std::endl;
        }
        std::cout << "epoch: " << epoch << " - last loss: " << outputs[0].scalar<float>() << std::endl;
    }

    std::vector<std::vector<float> >test_images;
    for(int i = 0; i < 100; i++){
        std::vector<float> image;
        for(int j = i * 784; j < i*784 + 784; j++){
            image.push_back(flat_images[j]);
        }
        test_images.push_back(image);
    }

    int i = 0;
    int correct = 0;
    for(auto image: test_images){
        Tensor x_test(DataTypeToEnum<float>::v(), 
                      TensorShape{1, static_cast<int>(images[0].size())});
        std::copy_n(image.begin(), image.size(), x_test.flat<float>().data());
        
        std::vector<Tensor> outputs_test;
        TF_CHECK_OK(session.Run({{x, x_test}}, {layer_2}, &outputs_test));
        int best = 0;
        for(int j = 0; j < 10; j++){
            if(outputs_test[0].flat<float>().data()[j] > outputs_test[0].flat<float>().data()[best]) best = j;
            std::cout << j << " " << outputs_test[0].flat<float>().data()[j];
        }
        std::cout << "\nCorrect: " << labels[i] << " - guess: " << best << std::endl;
        if(labels[i] == best)
            correct++;
        //mnist::print_image(images[i]);
        i++;
    }

    std::cout << "\n correct: " << correct << " / 100\n\n";
    return 0;
}
