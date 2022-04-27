//
// Created by aichao on 2022/4/22.
//

#ifndef FACEREC_NCNN_FACEDETECTOR_H
#define FACEREC_NCNN_FACEDETECTOR_H

#include <include/headers.h>
#include "include/types.h"

//
// Created by aichao on 2022/4/24.
//




class FaceDetector {
public:

    explicit FaceDetector(int _num_outputs = 9,
                          int _num_threads = 1,
                          int _input_height = 112,
                          int _input_width = 112);

    ~FaceDetector() = default;

private:
// nested classes
    typedef struct {
        float cx;
        float cy;
        float stride;
    } SCRFDPoint;
    typedef struct {
        float ratio;
        int dw;
        int dh;
        bool flag;
    } SCRFDScaleParams;

private:

    int num_threads = 1;
    int num_outputs = 6;
    int input_height = 320;
    int input_width = 320;
    std::shared_ptr<ncnn::Net> net;

    const float mean_vals[3] = {127.5f, 127.5f, 127.5f}; // RGB
    const float norm_vals[3] = {1.f / 128.f, 1.f / 128.f, 1.f / 128.f};

    unsigned int fmc = 3; // feature map count
    bool use_kps = false;
    unsigned int num_anchors = 2;
    std::vector<int> feat_stride_fpn = {8, 16, 32}; // steps, may [8, 16, 32, 64, 128]

    std::unordered_map<int, std::vector<SCRFDPoint>> center_points;
    bool center_points_is_update = false;
    static constexpr const unsigned int nms_pre = 1000;
    static constexpr const unsigned int max_nms = 30000;

private:

    void transform(const cv::Mat &mat_rs, ncnn::Mat &in);

    void initial_context();

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        SCRFDScaleParams &scale_params);

// generate once.
    void generate_points(int target_height, int target_width);

    void generate_bboxes_single_stride(const SCRFDScaleParams &scale_params,
                                       ncnn::Mat &score_pred,
                                       ncnn::Mat &bbox_pred,
                                       unsigned int stride,
                                       float score_threshold,
                                       float img_height,
                                       float img_width,
                                       std::vector<types::BoxfWithLandmarks> &bbox_kps_collection);

    void generate_bboxes_kps_single_stride(const SCRFDScaleParams &scale_params,
                                           ncnn::Mat &score_pred,
                                           ncnn::Mat &bbox_pred,
                                           ncnn::Mat &kps_pred,
                                           unsigned int stride,
                                           float score_threshold,
                                           float img_height,
                                           float img_width,
                                           std::vector<types::BoxfWithLandmarks> &bbox_kps_collection);

    void generate_bboxes_kps(const SCRFDScaleParams &scale_params,
                             std::vector<types::BoxfWithLandmarks> &bbox_kps_collection,
                             ncnn::Extractor &extractor,
                             float score_threshold, float img_height,
                             float img_width); // rescale & exclude

    void nms_bboxes_kps(std::vector<types::BoxfWithLandmarks> &input,
                        std::vector<types::BoxfWithLandmarks> &output,
                        float iou_threshold, unsigned int topk);

public:

    void load_model(const std::string &_param_path, const std::string &_bin_path);

    void detect(const cv::Mat &mat, std::vector<types::BoxfWithLandmarks> &detected_boxes_kps,
                float score_threshold = 0.25f, float iou_threshold = 0.45f,
                unsigned int topk = 400);

};


#endif //FACEREC_NCNN_FACEDETECTOR_H
