//
// Created by aichao on 2022/4/22.
//

#include "include/utils.h"

namespace utils {
    void blending_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                      float iou_threshold, unsigned int topk) {
        if (input.empty()) return;
        std::sort(input.begin(), input.end(),
                  [](const types::Boxf &a, const types::Boxf &b) { return a.score > b.score; });
        const unsigned int box_num = input.size();
        std::vector<int> merged(box_num, 0);

        unsigned int count = 0;
        for (unsigned int i = 0; i < box_num; ++i) {
            if (merged[i]) continue;
            std::vector<types::Boxf> buf;

            buf.push_back(input[i]);
            merged[i] = 1;

            for (unsigned int j = i + 1; j < box_num; ++j) {
                if (merged[j]) continue;

                float iou = static_cast<float>(input[i].iou_of(input[j]));
                if (iou > iou_threshold) {
                    merged[j] = 1;
                    buf.push_back(input[j]);
                }
            }

            float total = 0.f;
            for (unsigned int k = 0; k < buf.size(); ++k) {
                total += std::exp(buf[k].score);
            }
            types::Boxf rects;
            for (unsigned int l = 0; l < buf.size(); ++l) {
                float rate = std::exp(buf[l].score) / total;
                rects.x1 += buf[l].x1 * rate;
                rects.y1 += buf[l].y1 * rate;
                rects.x2 += buf[l].x2 * rate;
                rects.y2 += buf[l].y2 * rate;
                rects.score += buf[l].score * rate;
            }
            rects.flag = true;
            output.push_back(rects);

            // keep top k
            count += 1;
            if (count >= topk)
                break;
        }
    }

    void offset_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                    float iou_threshold, unsigned int topk) {
        if (input.empty()) return;
        std::sort(input.begin(), input.end(),
                  [](const types::Boxf &a, const types::Boxf &b) { return a.score > b.score; });
        const unsigned int box_num = input.size();
        std::vector<int> merged(box_num, 0);

        const float offset = 4096.f;
        /** Add offset according to classes.
         * That is, separate the boxes into categories, and each category performs its
         * own NMS operation. The same offset will be used for those predicted to be of
         * the same category. Therefore, the relative positions of boxes of the same
         * category will remain unchanged. Box of different classes will be farther away
         * after offset, because offsets are different. In this way, some overlapping but
         * different categories of entities are not filtered out by the NMS. Very clever!
         */
        for (unsigned int i = 0; i < box_num; ++i) {
            input[i].x1 += static_cast<float>(input[i].label) * offset;
            input[i].y1 += static_cast<float>(input[i].label) * offset;
            input[i].x2 += static_cast<float>(input[i].label) * offset;
            input[i].y2 += static_cast<float>(input[i].label) * offset;
        }

        unsigned int count = 0;
        for (unsigned int i = 0; i < box_num; ++i) {
            if (merged[i]) continue;
            std::vector<types::Boxf> buf;

            buf.push_back(input[i]);
            merged[i] = 1;

            for (unsigned int j = i + 1; j < box_num; ++j) {
                if (merged[j]) continue;

                float iou = static_cast<float>(input[i].iou_of(input[j]));

                if (iou > iou_threshold) {
                    merged[j] = 1;
                    buf.push_back(input[j]);
                }

            }
            output.push_back(buf[0]);

            // keep top k
            count += 1;
            if (count >= topk)
                break;
        }

        /** Substract offset.*/
        if (!output.empty()) {
            for (unsigned int i = 0; i < output.size(); ++i) {
                output[i].x1 -= static_cast<float>(output[i].label) * offset;
                output[i].y1 -= static_cast<float>(output[i].label) * offset;
                output[i].x2 -= static_cast<float>(output[i].label) * offset;
                output[i].y2 -= static_cast<float>(output[i].label) * offset;
            }
        }

    }

    void hard_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                  float iou_threshold, unsigned int topk) {
        if (input.empty()) return;
        std::sort(input.begin(), input.end(),
                  [](const types::Boxf &a, const types::Boxf &b) { return a.score > b.score; });
        const unsigned int box_num = input.size();
        std::vector<int> merged(box_num, 0);

        unsigned int count = 0;
        for (unsigned int i = 0; i < box_num; ++i) {
            if (merged[i]) continue;
            std::vector<types::Boxf> buf;

            buf.push_back(input[i]);
            merged[i] = 1;

            for (unsigned int j = i + 1; j < box_num; ++j) {
                if (merged[j]) continue;

                float iou = static_cast<float>(input[i].iou_of(input[j]));

                if (iou > iou_threshold) {
                    merged[j] = 1;
                    buf.push_back(input[j]);
                }

            }
            output.push_back(buf[0]);

            // keep top k
            count += 1;
            if (count >= topk)
                break;
        }
    }


    void draw_boxes_inplace(cv::Mat &mat_inplace, const std::vector<types::Boxf> &boxes) {
        if (boxes.empty()) return;
        for (const auto &box: boxes) {
            if (box.flag) {
                cv::rectangle(mat_inplace, box.rect(), cv::Scalar(255, 255, 0), 1);
                if (box.label_text) {
                    std::string label_text(box.label_text);
                    label_text += ":" + std::to_string(box.score).substr(0, 4);
                    cv::putText(mat_inplace, label_text, box.tl(), cv::FONT_HERSHEY_SIMPLEX,
                                0.6f, cv::Scalar(0, 255, 0), 1);

                }
            }
        }
    }

    void draw_boxes_with_landmarks_inplace(cv::Mat &mat_inplace, const std::vector<types::BoxfWithLandmarks> &boxes_kps,
                                           const std::string &text) {
        if (boxes_kps.empty()) return;
        for (const auto &box_kps: boxes_kps) {
            if (box_kps.flag) {
                // box
                if (box_kps.box.flag) {
                    cv::rectangle(mat_inplace, box_kps.box.rect(), cv::Scalar(255, 255, 0), 1);
                    if (!text.empty()) {
                        cv::putText(mat_inplace, text, box_kps.box.tl(), cv::FONT_HERSHEY_SIMPLEX,
                                    0.6f, cv::Scalar(0, 255, 0), 1);

                    }
                }
                // landmarks
                if (box_kps.landmarks.flag && !box_kps.landmarks.points.empty()) {
                    for (const auto &point: box_kps.landmarks.points)
                        cv::circle(mat_inplace, point, 2, cv::Scalar(0, 255, 0), -1);
                }
            }
        }
    }

    void clip_box(const cv::Mat &src_img, cv::Rect &box) {

        if (box.width > (src_img.cols - box.x)) {

            box.width = src_img.cols - box.x - 1;
        }
        if (box.height > (src_img.rows - box.y)) {

            box.height = src_img.rows - box.y - 1;
        }

    }


/*
使用五点人脸对齐
left_eye, right_eye, nose, left_month, right_month
*/
    cv::Mat face_align_bypoint5(cv::Mat &src_frame, std::vector<cv::Point2d> landmark5, std::vector<cv::Point2d> box) {
        Eigen::MatrixXf src_landmark_mtx(5, 2);
        int i = 0;
        for (int r = 0; r < landmark5.size(); r++) {
            src_landmark_mtx(r, 0) = landmark5.at(r).x;
            src_landmark_mtx(r, 1) = landmark5.at(r).y;
            i++;
        }

        Eigen::Matrix<float, 5, 2> dst_landmark_mtx;
        dst_landmark_mtx << 30.2946, 51.6963,
                65.5318, 51.6963,
                48.0252, 71.7366,
                33.5493, 92.3655,
                62.7299, 92.3655;

        int rows = src_landmark_mtx.rows();
        int cols = src_landmark_mtx.cols();
        Eigen::MatrixXf mean1 = src_landmark_mtx.colwise().mean(); // [1,2]
        Eigen::MatrixXf mean2 = dst_landmark_mtx.colwise().mean(); // [1,2]
        float col_mean_1 = mean1(0, 0);  // 列的均值
        float col_mean_2 = mean2(0, 0);

        //cout << src_landmark_mtx << endl;
        //cout << dst_landmark_mtx << endl;

        // std = sqrt(mean(abs(x - x.mean())**2))
        Eigen::Matrix<float, 5, 2> m1, m2;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                src_landmark_mtx(r, c) -= mean1(0, c);
                dst_landmark_mtx(r, c) -= mean2(0, c);

                auto abs_v1 = std::abs(src_landmark_mtx(r, c));
                auto abs_v2 = std::abs(dst_landmark_mtx(r, c));
                m1(r, c) = pow(abs_v1, 2);
                m2(r, c) = pow(abs_v2, 2);
            }
        }

        float std1 = sqrt(m1.mean());
        float std2 = sqrt(m2.mean());
        src_landmark_mtx /= std1;
        dst_landmark_mtx /= std2;


        Eigen::MatrixXf m = src_landmark_mtx.transpose() * dst_landmark_mtx;  // [2,2]
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::MatrixXf VT = svd.matrixV().transpose();
        Eigen::MatrixXf U = svd.matrixU();
        Eigen::MatrixXf A = svd.singularValues();
        Eigen::MatrixXf R = (U * VT).transpose();  // [2,2]


        Eigen::MatrixXf M(3, 3);
        M.row(2) << 0., 0., 1.;
        Eigen::MatrixXf t1 = (std1 / std2) * R;  // [2,2]
        Eigen::MatrixXf t2 = mean2.transpose() - (std1 / std2) * R * mean1.transpose();  // [2,1]
        M.row(0) << t1(0, 0), t1(0, 1), t2(0, 0);
        M.row(1) << t1(1, 0), t1(1, 1), t2(1, 0);

        //printf_s(" ============= transformer =============\n");
        //cout << M << endl;

        //printf_s(" ============= face transformer =============\n");
        cv::Mat cv_transformMat = (cv::Mat_<double>(2, 3) << \
             M(0, 0), M(0, 1), M(0, 2), \
             M(1, 0), M(1, 1), M(1, 2));
        cv::Mat align_img = cv::Mat::zeros(src_frame.rows * 3, src_frame.cols * 3, src_frame.type());
        src_frame.copyTo(align_img(cv::Rect(100, 100, src_frame.cols, src_frame.rows)));
        for (size_t k = 0; k < box.size(); k++) {
            box.at(k).x += 100;
            box.at(k).y += 100;
        }

        for (size_t k = 0; k < landmark5.size(); k++) {
            landmark5.at(k).x += 100;
            landmark5.at(k).y += 100;
//            circle(align_img, landmark5.at(k), 2, cv::Scalar(0, 255, 0), 1);
        }

        warpAffine(align_img, align_img, cv_transformMat, align_img.size());


        cv::Rect roi(0, 0, 0, 0);
        for (size_t r = 0; r < box.size(); r++) {
            auto x = box.at(r).x, y = box.at(r).y;
            box[r].x = x * M(0, 0) + y * M(0, 1) + M(0, 2);
            box[r].y = x * M(1, 0) + y * M(1, 1) + M(1, 2);
        }
        for (size_t r = 0; r < landmark5.size(); r++) {
            auto x = landmark5.at(r).x, y = landmark5.at(r).y;
            landmark5[r].x = x * M(0, 0) + y * M(0, 1) + M(0, 2);
            landmark5[r].y = x * M(1, 0) + y * M(1, 1) + M(1, 2);
        }
        // 由于box的点对齐后会出现 脸不全的情况，需要使用对齐后的关键点来矫正box
        auto dis_left_eye_right_eye = landmark5[1].x - landmark5[0].x;
        auto dis_left_month_left_eye = landmark5[landmark5.size() - 2].y - landmark5[0].y;
        box[0].x = std::min(box[0].x, landmark5[0].x - dis_left_eye_right_eye / 2);      // left top X;
        box[0].y = std::min(box[0].y, landmark5[0].y - dis_left_month_left_eye / 1.1);       // left top Y;
        box[box.size() - 1].x = std::max(box[box.size() - 1].x,
                                         landmark5[1].x + dis_left_eye_right_eye / 2);        // right bottom X;
        box[box.size() - 1].y = std::max(box[box.size() - 1].y, landmark5[landmark5.size() - 1].y +
                                                                dis_left_month_left_eye / 1.1);      // right bottom y;

        roi.x = std::max(0, int(box[0].x));
        roi.y = std::max(0, int(box[0].y));
        roi.width = std::min(int(box.at(box.size() - 1).x - roi.x) + 10, align_img.cols - roi.x);
        roi.height = std::min(int(box.at(box.size() - 1).y - roi.y) - 10, align_img.rows - roi.y - 10);
        cv::Mat dst;
        align_img(roi).copyTo(dst);
        return dst;
    }

    cv::Mat face_align_bypoint5(cv::Mat &src_frame, std::vector<cv::Point2f> &landmark5,
                                cv::Rect &box_) {
        std::vector<cv::Point2d> landmark;
        for (auto &point: landmark5) {
            landmark.push_back(point);
        }
        std::vector<cv::Point2d> box;
        box.push_back(cv::Point2d(box_.x, box_.y));
        box.push_back(cv::Point2d(box_.x + box_.width, box_.y + box_.height));
        cv::Mat dst = face_align_bypoint5(src_frame, landmark, box);
        return dst;
    }

    cv::Mat face_align_bypoint5(cv::Mat &src_frame, std::vector<types::BoxfWithLandmarks> detected_boxes_kps) {
        std::vector<cv::Point2f> landmark5 = detected_boxes_kps[0].landmarks.points;
        cv::Rect box = detected_boxes_kps[0].box.rect();
        return face_align_bypoint5(src_frame, landmark5, box);
    }
}