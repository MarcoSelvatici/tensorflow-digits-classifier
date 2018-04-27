#include "mnist-loader.hh"

namespace mnist{

int reverse_int(int i){
    unsigned char c1, c2, c3, c4;
    
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

std::vector<std::vector<int> > read_mnist_images(std::string file_name){
    std::vector<std::vector<int> >images;
    std::ifstream file(DATASET_PATH + file_name);
    if (file.is_open()){
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number)); 
        magic_number = reverse_int(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverse_int(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = reverse_int(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = reverse_int(n_cols);

        for(int i = 0; i < number_of_images; ++i){
            std::vector<int> image;
            for(int r = 0; r < n_rows; ++r){
                for(int c = 0; c < n_cols; ++c){
                    unsigned char temp = 0;
                    file.read((char*)&temp, sizeof(temp));
                    image.push_back((int)temp);
                }
            }
            images.push_back(image);
        }
    }else{
        std::cout << "ERROR: could not open mnist images file:" << std::endl
                  << "file " << file_name << " not found inside mnist-loader/" << std::endl;
    }
    return images;
}

std::vector<int> read_mnist_labels(std::string file_name){
    std::vector<int> labels;
    std::ifstream file(DATASET_PATH + file_name);
    if (file.is_open()){
        int magic_number = 0;
        int number_of_labels = 0;
        file.read((char*)&magic_number, sizeof(magic_number)); 
        magic_number = reverse_int(magic_number);
        file.read((char*)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverse_int(number_of_labels);

        for(int i = 0; i < number_of_labels; ++i){
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            labels.push_back((int)temp);
        }
    }else{
        std::cout << "ERROR: could not open mnist label file:" << std::endl
                  << "file " << file_name << " not found inside mnist-loader/" << std::endl;
    }
    return labels;
}

void print_image(std::vector<int> image){
    assert(image.size() == 784);

    for(int r = 0; r < 28; r++){
        for(int c = 0; c < 28; c++){
            if(image[r * 28 + c] < 100){
                std::cout << "#";
            }else{
                std::cout << " ";
            }
        }
        std::cout << std::endl;
    }
}

std::vector<int> get_train_labels(){
    return read_mnist_labels(TRAIN_LABELS_FILE_NAME);
}

std::vector<int> get_test_labels(){
    return read_mnist_labels(TEST_LABELS_FILE_NAME);
}

std::vector<std::vector<int> > get_train_images(){
    return read_mnist_images(TRAIN_IMAGES_FILE_NAME);
}

std::vector<std::vector<int> > get_test_images(){
    return read_mnist_images(TEST_IMAGES_FILE_NAME);
}

}