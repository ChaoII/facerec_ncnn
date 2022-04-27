
#include "include/utils.h"
#include "include/face_pipeline.h"


int main() {

    FacePipeline pipeline("../models/scrfd", "../models/facenet_casia-webface_resnet");
    pipeline.build_index("../facelib");
    cv::VideoCapture cap(1);
    cap.set(3, 640);
    cap.set(4, 480);
    cv::Mat frame;

    while (cap.isOpened()) {
        double st = ncnn::get_current_time();
        cap >> frame;
        std::vector<types::BoxfWithLandmarks> detected_boxes_kps;
        pipeline.get_face_det_obj()->detect(frame, detected_boxes_kps);
        if (!detected_boxes_kps.empty()) {
            cv::Mat face_img = utils::face_align_bypoint5(frame, detected_boxes_kps);;
            cv::imshow("123", face_img);
//            cv::Rect box = detected_boxes_kps[0].box.rect();
//            utils::clip_box(frame, box);
//            frame(box).copyTo(face_img);
            std::vector<float> feature;
            pipeline.get_face_rec_obj()->recognize(face_img, feature);
            std::string label = pipeline.get_label(feature);
            std::cout << label << std::endl;
            utils::draw_boxes_with_landmarks_inplace(frame, detected_boxes_kps, label);
        }
        double et = ncnn::get_current_time();
        std::cout << 1000 / (et - st) << std::endl;
        cv::imshow("13", frame);
        if (cv::waitKey(30) == 27) break;
    }


}