load("//tensorflow:tensorflow.bzl", "tf_cc_test")

cc_library(
    name = "mnistNet",
    hdrs = ["mnistNet.hh"],
    srcs = ["mnistNet.cc"],

    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
        "//tensorflow/cc:gradients",
        "//tensorflow/cc:grad_ops"
    ]
)

tf_cc_test(
    name = "digits_tester",
    srcs = ["digits_tester.cc"],

    deps = [
        "//tensorflow/cc/digits_classifier_prj/mnist-loader:mnist-loader",
        "//tensorflow/cc/digits_classifier_prj/digits_classifier:mnistNet"
    ]
)
