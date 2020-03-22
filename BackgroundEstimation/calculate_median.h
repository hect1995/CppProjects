//
// Created by Héctor Esteban Cabezos on 2020-03-22.
//

#ifndef BACKGROUNDESTIMATION_CALCULATE_MEDIAN_H
#define BACKGROUNDESTIMATION_CALCULATE_MEDIAN_H

#include <opencv2/opencv.hpp>
#include <iostream>

int computeMedian(std::vector<int> elements);


cv::Mat compute_median(std::vector<cv::Mat> vec);


#endif //BACKGROUNDESTIMATION_CALCULATE_MEDIAN_H
