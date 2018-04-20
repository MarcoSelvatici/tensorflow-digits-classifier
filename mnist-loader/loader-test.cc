#include "mnist-loader.hh"
#include <vector>
#include <iostream>

int main(){
    auto images = mnist::get_train_images();
    auto labels = mnist::get_train_labels();
    for(int i = 0; i < 2; i++){
        std::cout << (int)labels[i] << std::endl;
        mnist::print_image(images[i]);
    }

    images = mnist::get_test_images();
    labels = mnist::get_test_labels();
    for(int i = 0; i < 2; i++){
        std::cout << (int)labels[i] << std::endl;
        mnist::print_image(images[i]);
    }
    
    return 0;
}