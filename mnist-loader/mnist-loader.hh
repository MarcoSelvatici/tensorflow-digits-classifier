#ifndef __MNIST_LOADER_H
#define __MNIST_LOADER_H

#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <cassert>

namespace mnist{

const std::string DATASET_PATH = "tensorflow/cc/digits_classifier_prj/mnist-loader/";
const std::string TRAIN_IMAGES_FILE_NAME = "train-images-idx3-ubyte";
const std::string TRAIN_LABELS_FILE_NAME = "train-labels-idx1-ubyte";
const std::string TEST_IMAGES_FILE_NAME = "t10k-images-idx3-ubyte";
const std::string TEST_LABELS_FILE_NAME = "t10k-labels-idx1-ubyte"; 

int reverse_int(int i);

std::vector<std::vector<int> > read_mnist_images(std::string file_name);

std::vector<int> read_mnist_labels(std::string file_name);

void print_image(std::vector<int> image);

std::vector<int> get_train_labels();

std::vector<int> get_test_labels();

std::vector<std::vector<int> > get_train_images();

std::vector<std::vector<int> > get_test_images();

}

#endif