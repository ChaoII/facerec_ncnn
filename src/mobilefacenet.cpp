//
// Created by aichao on 2022/4/24.
//

#include "include/mobilefacenet.h"

NCNNMobileFaceNet::NCNNMobileFaceNet(const std::string &_param_path,
                                     const std::string &_bin_path,
                                     unsigned int _num_threads) {
    net = std::make_shared<ncnn::Net>();
//    net->opt.use_vulkan_compute = true;
    net->opt.num_threads = _num_threads;
    net->load_param(_param_path.c_str());
    net->load_model(_bin_path.c_str());

};

void NCNNMobileFaceNet::transform(const cv::Mat &mat, ncnn::Mat &in) {
    // BGR NHWC -> RGB NCHW
    int h = mat.rows;
    int w = mat.cols;
    in = ncnn::Mat::from_pixels_resize(
            mat.data, ncnn::Mat::PIXEL_BGR2RGB,
            w, h, input_width, input_height
    );
    in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNNMobileFaceNet::detect(const cv::Mat &mat, types::FaceContent &face_content) {
    if (mat.empty()) return;
    // 1. make input tensor
    ncnn::Mat input;
    this->transform(mat, input);
    // 2. inference & extract
    auto extractor = net->create_extractor();
    extractor.input("input", input);
    ncnn::Mat embedding;
    extractor.extract("embedding", embedding);

    const unsigned int hidden_dim = embedding.w; // 512
    const float *embedding_values = (float *) embedding.data;
    std::vector<float> embedding_norm(embedding_values, embedding_values + hidden_dim);
    cv::normalize(embedding_norm, embedding_norm); // l2 normalize
    face_content.embedding.assign(embedding_norm.begin(), embedding_norm.end());
    face_content.dim = hidden_dim;
    face_content.flag = true;
}
