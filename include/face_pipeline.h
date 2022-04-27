//
// Created by aichao on 2022/4/22.
//

#ifndef FACEREC_NCNN_FACE_PIPELINE_H
#define FACEREC_NCNN_FACE_PIPELINE_H

#include "include/FaceDetector.h"
#include "include/FaceRecognition.h"
#include "include/utils.h"
#include <fstream>

class FacePipeline {
    struct face_lib {
        std::string name;
        std::vector<float> feature;
    };
public:
    FacePipeline(const std::string &det_model_dir, const std::string &rec_model_dir);

    void build_index(const std::string &img_dir);

    std::string get_label(std::vector<float> &cur_feature);

    std::shared_ptr<FaceDetector> get_face_det_obj();

    std::shared_ptr<FaceRecognition> get_face_rec_obj();

private:
    float threshold = 0.45;
    std::vector<face_lib> face_libs;
    std::shared_ptr<FaceDetector> face_det = nullptr;
    std::shared_ptr<FaceRecognition> face_rec = nullptr;
};


#endif //FACEREC_NCNN_FACE_PIPELINE_H
