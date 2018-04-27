#include "tensorflow/cc/digits_classifier_prj/digits_classifier/mnistNet.hh"

MnistNet::MnistNet() : scope(Scope::NewRootScope()) {
    std::cout << "Creating net graph..." << std::endl;
    std::srand(std::time(NULL));

    scope = Scope::NewRootScope();
    x = new Placeholder(scope, DT_FLOAT);
    y = new Placeholder(scope, DT_FLOAT);

    // weights and biases
    w1 = new Variable(scope, {size_input, size_first_layer}, DT_FLOAT);
    assign_w1 = new Assign(scope, *w1, RandomNormal(scope, {size_input, size_first_layer}, DT_FLOAT));
    
    w2 = new Variable(scope, {size_first_layer, size_output}, DT_FLOAT);
    assign_w2 = new Assign(scope, *w2, RandomNormal(scope, {size_first_layer, size_output}, DT_FLOAT));

    b1 = new Variable(scope, {1, size_first_layer}, DT_FLOAT);
    assign_b1 = new Assign(scope, *b1, RandomNormal(scope, {1, size_first_layer}, DT_FLOAT));

    b2 = new Variable(scope, {1, size_output}, DT_FLOAT);
    assign_b2 = new Assign(scope, *b2, RandomNormal(scope, {1, size_output}, DT_FLOAT));

    // activation for the layers
    layer_1 = new Sigmoid(scope, Add(scope, MatMul(scope, *x, *w1), *b1));
    layer_2 = new Sigmoid(scope, Add(scope, MatMul(scope, *layer_1, *w2), *b2));

    // loss
    loss = new ReduceMean(scope, Square(scope, Subtract(scope, *layer_2, *y)), {0, 1});

    // prepare for gradient descent
    std::vector<Output> grad_outputs;
    TF_CHECK_OK(AddSymbolicGradients(scope, {*loss}, {*w1, *w2, *b1, *b2}, &grad_outputs));

    // update the weights and bias using gradient descent
    apply_w1 = new ApplyGradientDescent(scope, *w1, Cast(scope, default_learning_rate,  DT_FLOAT), {grad_outputs[0]});
    apply_w2 = new ApplyGradientDescent(scope, *w2, Cast(scope, default_learning_rate,  DT_FLOAT), {grad_outputs[1]});
    apply_b1 = new ApplyGradientDescent(scope, *b1, Cast(scope, default_learning_rate,  DT_FLOAT), {grad_outputs[2]});
    apply_b2 = new ApplyGradientDescent(scope, *b2, Cast(scope, default_learning_rate,  DT_FLOAT), {grad_outputs[3]});

    // create session to run the graph
    session = new ClientSession(scope);
}

std::vector<float> MnistNet::get_flat_images(std::vector< std::vector<int> >images){
    std::vector<float> flat_images;
    for(std::vector<int> image: images){
        std::vector<float> f_image(image.begin(), image.end());
        flat_images.insert(flat_images.end(), f_image.begin(), f_image.end());
    }
    return flat_images;
}

std::vector<float> MnistNet::get_flat_one_hot_labels(std::vector<int> labels){
    std::vector<float> flat_one_hot_labels;
    for(int label: labels){
        std::vector<float> tmp(10, 0.0f);
        tmp[label] = 1.0f;
        flat_one_hot_labels.insert(flat_one_hot_labels.end(), tmp.begin(), tmp.end());
    }
    return flat_one_hot_labels;
}

void MnistNet::train(std::vector< std::vector<int> >images,
                     std::vector<int> labels,
                     int number_of_epochs){
    std::cout << "Training the net..." << std::endl;
    if(number_of_epochs == NULL){
        number_of_epochs = default_number_of_epochs;
    }

    // get flatten images of floats
    std::vector<float> flat_images = get_flat_images(images);
    // get flatten, one-hot labels of floats
    std::vector<float> flat_one_hot_labels = get_flat_one_hot_labels(labels);

    // create session to run the graph
    // ClientSession session(scope);

    // init the weights and biases by running the assigns nodes once
    TF_CHECK_OK(session->Run({*assign_w1, *assign_w2, *assign_b1, *assign_b2}, nullptr));

    // training
    for(int epoch = 0; epoch < number_of_epochs; epoch++){
        float minibatch_total_loss = 0;
        for(int minibatch = 0; minibatch < images.size() / minibatch_size; minibatch++){
            std::vector<float> minibatch_images;
            std::vector<float> minibatch_labels;
            minibatch_images.reserve(minibatch_size * image_size);
            minibatch_labels.reserve(minibatch_size * label_size);
            // select a bunch of random samples for the minibatch
            for(int sample = 0; sample < minibatch_size; sample++){
                int idx = std::rand() % images.size();
                std::copy(begin(flat_images) + idx * image_size,
                          begin(flat_images) + idx * image_size + image_size,
                          std::back_inserter(minibatch_images));
                std::copy(begin(flat_one_hot_labels) + idx * label_size,
                          begin(flat_one_hot_labels) + idx * label_size + label_size,
                          std::back_inserter(minibatch_labels));
            }
            // create the tensors for the mini_batch to feed the net with
            Tensor x_minibatch(DataTypeToEnum<float>::v(), 
                               TensorShape{static_cast<int>(minibatch_size), static_cast<int>(image_size)});
            std::copy_n(minibatch_images.begin(), minibatch_images.size(), x_minibatch.flat<float>().data());
            Tensor y_minibatch(DataTypeToEnum<float>::v(), 
                               TensorShape{static_cast<int>(minibatch_size), label_size});
            std::copy_n(minibatch_labels.begin(), minibatch_labels.size(), y_minibatch.flat<float>().data());

            std::vector<Tensor> outputs;
            // calculate the loss for the minibatch
            TF_CHECK_OK(session->Run({{*x, x_minibatch}, {*y, y_minibatch}}, {*loss}, &outputs));
            // backpropagation (apply gradients)
            TF_CHECK_OK(session->Run({{*x, x_minibatch}, {*y, y_minibatch}},
                                     {*apply_w1, *apply_w2, *apply_b1, *apply_b2, *layer_2}, nullptr));

            minibatch_total_loss += *outputs[0].scalar<float>().data();
            // std::cout << "  minibatch: " << minibatch << " - loss: " << *outputs[0].scalar<float>().data() << std::endl;
        }
        std::cout << "epoch: " << epoch << " - avg loss: " << minibatch_total_loss / (float)(images.size() / minibatch_size) << std::endl;
    }
}

std::vector<int> MnistNet::predict(std::vector<std::vector<int> >images){
    std::cout << "Predicting..." << std::endl;

    int num_of_images = images.size();
    std::vector<float> flat_images = get_flat_images(images);

    // copy the images to a tensor to fit the net with
    Tensor x_test(DataTypeToEnum<float>::v(),
                  TensorShape{static_cast<int>(num_of_images), static_cast<int>(image_size)});
    std::copy_n(flat_images.begin(), flat_images.size(), x_test.flat<float>().data());

    // predict
    std::vector<Tensor> outputs_test;
    TF_CHECK_OK(session->Run({{*x, x_test}}, {*layer_2}, &outputs_test));

    // extract results
    std::vector<int> predictions;
    for(int prediction = 0; prediction < num_of_images; prediction++){
        // find the best prediction
        int best = 0;
        auto activations = outputs_test[0].flat<float>();
        for(int i = 1; i < 10; i++){
            if(activations.data()[i + prediction * 10] > activations.data()[best + prediction * 10]){
                best = i;
            }
        }
        predictions.push_back(best);
    }
    return predictions;
}
