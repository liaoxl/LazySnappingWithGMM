#ifndef LSAPPLICATION_H
#define LSAPPLICATION_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "gmm.h"
#include "lazysnapping.h"

using namespace cv;

#define UN_KNOWN 5

const Scalar RED = Scalar(0, 0, 255);
const Scalar PINK = Scalar(230, 130, 255);
const Scalar BLUE = Scalar(255, 0, 0);
const Scalar LIGHTBLUE = Scalar(255, 255, 160);

class LSApplication{
public:
    enum{ NOT_SET = 0, IN_PROCESS = 1, SET = 2 };

    static const int BGD_KEY = CV_EVENT_FLAG_CTRLKEY;
    static const int FGD_KEY = CV_EVENT_FLAG_SHIFTKEY;

    static const int radius = 2;
    static const int thickness = -1;

    void reset();
    void setImageAndWinName( const Mat& _image, const string& _winName );
    void showImage() const;
    void mouseClick( int event, int x, int y, int flags, void* param );

private:
    void setLblsInMask( int flags, Point p);

    const string* winName;
    const Mat* image;
    Mat mask;
    Mat bgdModel, fgdModel;

    uchar lblsState;
    bool isInitialized;

    vector<Point> fgdPxls, bgdPxls;
};

void LSApplication::reset()
{
    if( !mask.empty() )
        mask.setTo(Scalar::all(UN_KNOWN));
    bgdPxls.clear(); fgdPxls.clear();

    isInitialized = false;
    lblsState = NOT_SET;
}

void LSApplication::setImageAndWinName( const Mat& _image, const string& _winName  )
{
    if( _image.empty() || _winName.empty() )
        return;
    image = &_image;
    winName = &_winName;
    mask.create( image->size(), CV_8UC1);
    reset();
}

void LSApplication::showImage() const
{
    if (image->empty() || winName->empty())
        return;

    Mat res;

    image->copyTo(res);

    Point p;
    for(p.y=0; p.y<res.rows; p.y++)
    {
        for(p.x=0; p.x<res.cols; p.x++)
        {
            if(mask.at<uchar>(p) == GC_PR_FGD)
            {
                circle(res, p, radius, PINK, thickness);
            }
            else if(mask.at<uchar>(p) == GC_PR_BGD)
            {
                circle(res, p, radius, LIGHTBLUE, thickness);
            }
        }
    }

    vector<Point>::const_iterator it;
    for (it = bgdPxls.begin(); it != bgdPxls.end(); ++it)
        circle(res, *it, radius, BLUE, thickness);
    for (it = fgdPxls.begin(); it != fgdPxls.end(); ++it)
        circle(res, *it, radius, RED, thickness);

    imshow(*winName, res);
}


void LSApplication::setLblsInMask( int flags, Point p )
{
    vector<Point> *bpxls, *fpxls;
    uchar bvalue, fvalue;

    bpxls = &bgdPxls;
    fpxls = &fgdPxls;
    bvalue = GC_BGD;
    fvalue = GC_FGD;

    if (flags & BGD_KEY) {
        bpxls->push_back(p);
        circle(mask, p, radius, bvalue, thickness);
    }
    if (flags & FGD_KEY) {
        fpxls->push_back(p);
        circle(mask, p, radius, fvalue, thickness);
    }

}

void LSApplication::mouseClick( int event, int x, int y, int flags, void* )
{
    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN: // set rect or GC_BGD(GC_FGD) labels
        {
            bool isb = (flags & BGD_KEY) != 0,
                 isf = (flags & FGD_KEY) != 0;
            if ( isb || isf )
                lblsState = IN_PROCESS;
        }
        break;
    case CV_EVENT_LBUTTONUP:
        if( lblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y));
            lblsState = SET;
            lazySnapping(*image, mask, bgdModel, fgdModel);
            showImage();
        }
        break;
    case CV_EVENT_MOUSEMOVE:
      if( lblsState == IN_PROCESS )
        {
            setLblsInMask(flags, Point(x,y));
            showImage();
        }
        break;
    }
}


#endif // LSAPPLICATION_H
