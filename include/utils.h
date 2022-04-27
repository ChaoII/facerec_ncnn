//
// Created by aichao on 2022/4/22.
//

#ifndef FACEREC_NCNN_UTILS_H
#define FACEREC_NCNN_UTILS_H

#include "include/types.h"
#include "eigen3/Eigen/Dense"


namespace utils {
    void blending_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                      float iou_threshold, unsigned int topk);

    void offset_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                    float iou_threshold, unsigned int topk);

    void hard_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                  float iou_threshold, unsigned int topk);

    void draw_boxes_inplace(cv::Mat &mat_inplace, const std::vector<types::Boxf> &boxes);

    void draw_boxes_with_landmarks_inplace(cv::Mat &mat_inplace, const std::vector<types::BoxfWithLandmarks> &boxes_kps,
                                           const std::string &text);

    void clip_box(const cv::Mat &src_img, cv::Rect &box);

    cv::Mat face_align_bypoint5(cv::Mat &src_frame, std::vector<cv::Point2d> landmark5,
                                std::vector<cv::Point2d> box);

    cv::Mat face_align_bypoint5(cv::Mat &src_frame, std::vector<cv::Point2f> &landmark5,
                                cv::Rect &box_);

    cv::Mat face_align_bypoint5(cv::Mat &src_frame, std::vector<types::BoxfWithLandmarks> detected_boxes_kps);

};


#endif //FACEREC_NCNN_UTILS_H
