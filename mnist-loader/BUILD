cc_library(
    name = "mnist-loader",
    srcs = ["mnist-loader.cc"],
    hdrs = ["mnist-loader.hh"],

    data = [
        ":t10k-images-idx3-ubyte",
        ":t10k-labels-idx1-ubyte",
        ":train-images-idx3-ubyte",
        ":train-labels-idx1-ubyte",
    ],

    visibility = [
        "//tensorflow/cc/digits_classifier_prj/digits_classifier:__pkg__"
    ],
)

cc_test(
    name = "loader-test",
    srcs = ["loader-test.cc"],

    deps = [
        ":mnist-loader"
    ]
)