//
// Created by aichao on 2022/4/22.
//

#ifndef FACEREC_NCNN_FACERECOGNITION_H
#define FACEREC_NCNN_FACERECOGNITION_H

#include "include/headers.h"

class FaceRecognition {

public:
    FaceRecognition() = default;

    void load_model(const std::string &param_path, const std::string &model_path);

    void recognize(cv::Mat &img, std::vector<float> &feature);

    template<typename T>
    float cosine_similarity(const std::vector<T> &a, const std::vector<T> &b) {
        float zero_vale = 0.f;
        if (a.empty() || b.empty() || (a.size() != b.size())) return zero_vale;
        const unsigned int _size = a.size();
        float mul_a = zero_vale, mul_b = zero_vale, mul_ab = zero_vale;
        for (unsigned int i = 0; i < _size; ++i) {
            mul_a += (float) a[i] * (float) a[i];
            mul_b += (float) b[i] * (float) b[i];
            mul_ab += (float) a[i] * (float) b[i];
        }
        if (mul_a == zero_vale || mul_b == zero_vale) return zero_vale;
        return (mul_ab / (std::sqrt(mul_a) * std::sqrt(mul_b)));
    }


private:
    std::shared_ptr<ncnn::Net> net = nullptr;
//    float mean_vals[3] = {0.5 * 255, 0.5 * 255, 0.5 * 255};
//    float norm_vals[3] = {1 / (0.5 * 255), 1 / (0.5 * 255), 1 / (0.5 * 255)};
//    int target_width = 112;
//    int target_height = 112;
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f}; // RGB
    const float norm_vals[3] = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
    static constexpr const int target_width = 112;
    static constexpr const int target_height = 112;
};


#endif //FACEREC_NCNN_FACERECOGNITION_H
