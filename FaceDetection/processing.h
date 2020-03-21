//
// Created by Héctor Esteban Cabezos on 2020-03-21.
//

#ifndef OPENCV_PROCESSING_H
#define OPENCV_PROCESSING_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/core/types.hpp>

void detectAndMark( cv::Mat& img, cv::CascadeClassifier& cascade,
                    cv::CascadeClassifier& nestedCascade);

#endif //OPENCV_PROCESSING_H
