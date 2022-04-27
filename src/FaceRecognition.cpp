//
// Created by aichao on 2022/4/22.
//

#include "include/FaceRecognition.h"


void FaceRecognition::load_model(const std::string &param_path, const std::string &model_path) {

    net = std::make_shared<ncnn::Net>();
//    net->opt.num_threads = ncnn::get_cpu_count();
//    net->opt.lightmode=false;
    net->load_param(param_path.c_str());
    net->load_model(model_path.c_str());
}

void FaceRecognition::recognize(cv::Mat &img, std::vector<float> &feature) {

    ncnn::Mat in;
    int h = img.rows;
    int w = img.cols;
    in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB,
                                       w, h, target_width, target_height);

    in.substract_mean_normalize(mean_vals, norm_vals);
    auto extractor = net->create_extractor();


    extractor.input("input", in);
    ncnn::Mat embedding;


    extractor.extract("embedding", embedding);

    const unsigned int feature_length = embedding.w; // 128
    const float *embedding_values = (float *) embedding.data;
    std::vector<float> embedding_norm(embedding_values, embedding_values + feature_length);
    cv::normalize(embedding_norm, embedding_norm); // l2 normalize
    feature.assign(embedding_norm.begin(), embedding_norm.end());
}

