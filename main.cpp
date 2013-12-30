#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "lsapplication.h"

using namespace std;
using namespace cv;

LSApplication lsapp;

void on_mouse(int event, int x, int y, int flags, void* param) {
    lsapp.mouseClick(event, x, y, flags, param);
}

int main()
{
    Mat image=imread("moondark.jpg");

    const string winName = "LazySnapping";
    cvNamedWindow(winName.c_str(), CV_WINDOW_NORMAL);
    cvSetMouseCallback(winName.c_str(), on_mouse, 0);

    lsapp.setImageAndWinName(image, winName);
    lsapp.showImage();


    for(;;)
    {
        int c = cvWaitKey(0);
        switch( (char) c )
        {
        case '\x1b':
            cout << "Exiting ..." << endl;
            goto exit_main;
        }
    }
exit_main:
    cvDestroyWindow( winName.c_str() );
    return 0;
}

