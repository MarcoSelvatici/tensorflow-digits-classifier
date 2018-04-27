#include <iostream>
#include <vector>

#include "tensorflow/cc/digits_classifier_prj/digits_classifier/mnistNet.hh"
#include "tensorflow/cc/digits_classifier_prj/mnist-loader/mnist-loader.hh"

int main(){
    auto train_images = mnist::get_train_images();
    auto train_labels = mnist::get_train_labels();
    auto test_images = mnist::get_test_images();
    auto test_labels = mnist::get_test_labels();

    MnistNet nn;
    nn.train(train_images, train_labels, 50);
    std::vector<int> predictions = nn.predict(test_images);
    
    int correct = 0;
    for(int i = 0; i < predictions.size(); i++){
        if(predictions[i] == test_labels[i]){
            correct++;
        }
    }
    std::cout << "Correct: " << correct << " / " << predictions.size() << std::endl; 
}