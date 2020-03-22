#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <unistd.h>
#include "calculate_median.h"

using namespace cv;

int main(int argc, char const *argv[])
{
    // Open camera
    VideoCapture capture;
    capture.open(0);

    if(!capture.isOpened())
        std::cerr << "Error opening video file\n";

    //Select 25 frames to compute the background
    std::vector<Mat> frames;
    Mat frame;
    int index = 0;
    int counter = 0;
    while(counter<25)
    {
        capture.read(frame);
        if(frame.empty())
            continue;
        if (index%50==0)
        {
            frames.push_back(frame);
            counter ++;
        }
    }
    // Calculate the median along the time axis
    Mat medianFrame = compute_median(frames);

    capture.set(CAP_PROP_POS_FRAMES, 0);

    // Convert background to grayscale
    Mat grayMedianFrame;
    cvtColor(medianFrame, grayMedianFrame, COLOR_BGR2GRAY);

    while(1)
    {
        // Read frame
        capture.read(frame);

        if (frame.empty())
            break;

        // Convert current frame to grayscale
        cvtColor(frame, frame, COLOR_BGR2GRAY);

        // Calculate absolute difference of current frame and the median frame
        Mat dframe;
        absdiff(frame, grayMedianFrame, dframe);

        // Threshold to binarize
        threshold(dframe, dframe, 35, 255, THRESH_BINARY);

        // Display Image
        imshow("frame", dframe);
        waitKey(20);
    }
    capture.release();
    return 1;

}