//
// Created by aichao on 2022/4/22.
//

#include "include/face_pipeline.h"

FacePipeline::FacePipeline(const std::string &det_model_dir, const std::string &rec_model_dir) {

    face_det = std::make_shared<FaceDetector>();
    face_rec = std::make_shared<FaceRecognition>();
    face_det->load_model(det_model_dir + "/model.param", det_model_dir + "/model.bin");
    face_rec->load_model(rec_model_dir + "/model.param", rec_model_dir + "/model.bin");
}

void FacePipeline::build_index(const std::string &img_dir) {
    std::fstream f(img_dir + "/facelib.txt");
    std::vector<std::string> file_names;
    std::string line;
    while (getline(f, line)) {
        file_names.push_back(line);
    }
    for (auto &file_name: file_names) {
        std::string img_path = img_dir + "/" + file_name;
        std::string label = file_name.substr(0, 6);
        std::cout << label << std::endl;
        cv::Mat img = cv::imread(img_path);
        std::vector<types::BoxfWithLandmarks> detected_boxes_kps;
        face_det->detect(img, detected_boxes_kps);
        cv::Mat dist = utils::face_align_bypoint5(img, detected_boxes_kps);
//        cv::Rect box = detected_boxes_kps[0].box.rect();
//        utils::clip_box(img, box);
//        img(box).copyTo(dist);
        std::vector<float> feature;
        face_rec->recognize(dist, feature);
        face_libs.push_back({label, feature});
    }
}

std::string FacePipeline::get_label(std::vector<float> &cur_feature) {
    float max_sim = 0;
    std::string label;
    for (auto &face_lib: face_libs) {
        float similarity = face_rec->cosine_similarity<float>(cur_feature, face_lib.feature);
        if (similarity > max_sim) {
            max_sim = similarity;
            label = face_lib.name;
        }
    }
    if (max_sim > threshold) {
        return label + ": " + std::to_string(max_sim).substr(0, 4);
    } else {
        return "unknow: " + std::to_string(max_sim).substr(0, 4);
    }
}

std::shared_ptr<FaceDetector> FacePipeline::get_face_det_obj() {
    return face_det;
}

std::shared_ptr<FaceRecognition> FacePipeline::get_face_rec_obj() {
    return face_rec;
}


