#include <iostream>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "processing.h"

using namespace cv;


int main( int argc, const char** argv )
{
    // Haar Model for the face
    String face_cascade_name = "models/haarcascade_frontalface_alt.xml";
    // Haar Model for the eyes
    String eyes_cascade_name = "models/haarcascade_eye_tree_eyeglasses.xml";

    //-- 1. Load the cascades
    CascadeClassifier face_cascade;
    CascadeClassifier eyes_cascade;
    if( !face_cascade.load( face_cascade_name ) )
    {
        std::cout << "--(!)Error loading face cascade\n";
        return -1;
    };
    if( !eyes_cascade.load( eyes_cascade_name ) )
    {
        std::cout << "--(!)Error loading eyes cascade\n";
        return -1;
    };
    VideoCapture capture;
    //-- 2. Read the video stream through the default camera
    capture.open(0);
    if ( ! capture.isOpened() )
    {
        std::cout << "--(!)Error opening video capture\n";
        return -1;
    }
    Mat frame;
    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            std::cout << "--(!) No captured frame -- Break!\n";
            break;
        }
        //-- 3. Apply the classifier to the frame
        detectAndMark( frame, face_cascade,eyes_cascade);
        if( waitKey(10) == 27 )
        {
            break; // escape
        }
    }
    return 0;
}