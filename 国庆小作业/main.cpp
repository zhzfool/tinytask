#include "windmill.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace std;
using namespace cv;

int main()
{
    // 初始化时间和 WindMill 类
    std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
    WINDMILL::WindMill wm(t.count());
    cv::Mat src;

    // 读取模板图像R和锤子图像
    cv::Mat target_img_R = imread("/home/zhouhongjin/第二次任务喵/task/image/R.png", IMREAD_GRAYSCALE);
    cv::Mat target_img_hammer = imread("/home/zhouhongjin/第二次任务喵/task/image/target.png", IMREAD_GRAYSCALE);

    if (target_img_R.empty() || target_img_hammer.empty()) {
        cout << "无法读取目标图像R或锤子图像，请检查文件路径是否正确。" << endl;
        return -1;
    }

    // 初始化 SIFT 特征检测器
    Ptr<SIFT> sift = SIFT::create();

    // 检测目标图像R的关键点和计算描述子
    vector<KeyPoint> keypoints_R;
    Mat descriptors_R;
    sift->detectAndCompute(target_img_R, noArray(), keypoints_R, descriptors_R);

    // 检测锤子图像的关键点和计算描述子
    vector<KeyPoint> keypoints_hammer;
    Mat descriptors_hammer;
    sift->detectAndCompute(target_img_hammer, noArray(), keypoints_hammer, descriptors_hammer);

    if (descriptors_R.empty() || descriptors_hammer.empty()) {
        cout << "无法检测到目标图像R或锤子图像的关键点或描述子。" << endl;
        return -1;
    }

    // 初始化计数器
    int total_frames = 0;
    int successful_R_counts = 0;
    int successful_hammer_counts = 0;

    while (1)
    {
        t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        src = wm.getMat((double)t.count() / 1000);

        // 处理帧计数器
        total_frames++;

        // 将实时获取的风车图像转换为灰度图
        Mat scene_img;
        cvtColor(src, scene_img, COLOR_BGR2GRAY);

        // 检测风车图像的关键点和计算描述子
        vector<KeyPoint> keypoints_scene;
        Mat descriptors_scene;
        sift->detectAndCompute(scene_img, noArray(), keypoints_scene, descriptors_scene);

        if (descriptors_scene.empty()) {
            cout << "无法检测到风车图像的关键点或描述子。" << endl;
            continue;
        }

        // 使用 FLANN 匹配器进行描述子匹配
        FlannBasedMatcher matcher(new flann::KDTreeIndexParams(5), new flann::SearchParams(50));
        vector<vector<DMatch>> knn_matches_R, knn_matches_hammer;

        // 匹配
        matcher.knnMatch(descriptors_R, descriptors_scene, knn_matches_R, 2);
        matcher.knnMatch(descriptors_hammer, descriptors_scene, knn_matches_hammer, 2);

        // 优化匹配
        vector<DMatch> good_matches_R, good_matches_hammer;
        for (size_t i = 0; i < knn_matches_R.size(); i++) {
            if (knn_matches_R[i][0].distance < 0.7 * knn_matches_R[i][1].distance) {
                good_matches_R.push_back(knn_matches_R[i][0]);
            }
        }

        for (size_t i = 0; i < knn_matches_hammer.size(); i++) {
            if (knn_matches_hammer[i][0].distance < 0.7 * knn_matches_hammer[i][1].distance) {
                good_matches_hammer.push_back(knn_matches_hammer[i][0]);
            }
        }

        // 检查匹配结果数量并绘制
        if (good_matches_R.size() >= 4) {
            vector<Point2f> src_pts_R;
            vector<Point2f> dst_pts_R;
            for (size_t i = 0; i < good_matches_R.size(); i++) {
                src_pts_R.push_back(keypoints_R[good_matches_R[i].queryIdx].pt);
                dst_pts_R.push_back(keypoints_scene[good_matches_R[i].trainIdx].pt);
            }

            // 使用RANSAC精确地计算变换矩阵
            Mat mask_R;
            Mat M_R = findHomography(src_pts_R, dst_pts_R, RANSAC, 3.0, mask_R);

            if (!M_R.empty()) {
                int h_R = target_img_R.rows;
                int w_R = target_img_R.cols;
                Point2f center_R(w_R / 2.0f, h_R / 2.0f);
                vector<Point2f> center_R_transformed(1);
                perspectiveTransform(vector<Point2f>{center_R}, center_R_transformed, M_R);
                circle(src, center_R_transformed[0], std::min(w_R, h_R) / 2, Scalar(0, 255, 0), 3, LINE_AA);

                // 成功识别 R 的计数
                successful_R_counts++;
            }
        }

        if (good_matches_hammer.size() >= 4) {
            vector<Point2f> src_pts_hammer;
            vector<Point2f> dst_pts_hammer;
            for (size_t i = 0; i < good_matches_hammer.size(); i++) {
                src_pts_hammer.push_back(keypoints_hammer[good_matches_hammer[i].queryIdx].pt);
                dst_pts_hammer.push_back(keypoints_scene[good_matches_hammer[i].trainIdx].pt);
            }

            Mat mask_hammer;
            Mat M_hammer = findHomography(src_pts_hammer, dst_pts_hammer, RANSAC, 5.0, mask_hammer);

            if (!M_hammer.empty()) {
                int h_hammer = target_img_hammer.rows;
                int w_hammer = target_img_hammer.cols;
                int head_center_y = static_cast<int>(h_hammer / 3.4);
                Point2f hammer_head_center(w_hammer / 2.0f, head_center_y);

                vector<Point2f> hammer_center_transformed(1);
                perspectiveTransform(vector<Point2f>{hammer_head_center}, hammer_center_transformed, M_hammer);
                Point2f hammer_center_point = hammer_center_transformed[0];

                // 绘制圆心在对称轴上的标记点
                circle(src, hammer_center_point, 12, Scalar(0, 255, 0), 2, LINE_AA);

                // 成功识别锤子的计数
                successful_hammer_counts++;
            }
        }

        // 显示结果图像
        imshow("Matched Image", src);

        // 检查退出条件
        if (waitKey(1) >= 0) {
            cout << "成功识别 R 的次数: " << successful_R_counts << endl;
            cout << "成功识别锤子的次数: " << successful_hammer_counts << endl;
            cout << "总处理帧数: " << total_frames << endl; // 输出总帧数
            break; // 退出循环
        }
    }

    return 0;
}
