//
// Created by aichao on 2022/4/24.
//

#ifndef FACEREC_NCNN_MOBILEFACENET_H
#define FACEREC_NCNN_MOBILEFACENET_H


#include "include/headers.h"
#include "include/types.h"


class NCNNMobileFaceNet {
public:
    explicit NCNNMobileFaceNet(const std::string &_param_path,
                               const std::string &_bin_path,
                               unsigned int _num_threads = 1);

    ~NCNNMobileFaceNet() = default;

private:
    std::shared_ptr<ncnn::Net> net = nullptr;
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f}; // RGB
    const float norm_vals[3] = {1.f / 128.0f, 1.f / 128.0f, 1.f / 128.0f};
    static constexpr const int input_width = 112;
    static constexpr const int input_height = 112;

private:

    void transform(const cv::Mat &mat, ncnn::Mat &in);


public:

    void detect(const cv::Mat &mat, types::FaceContent &face_content);
};


#endif //FACEREC_NCNN_MOBILEFACENET_H
