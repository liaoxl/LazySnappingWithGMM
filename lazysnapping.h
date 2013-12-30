#ifndef LAZYSNAPPING_H
#define LAZYSNAPPING_H

#include "gmm.h"
#include <opencv2/highgui/highgui.hpp>
#include "maxflow-v3.01/graph.h"

using namespace cv;

typedef Graph<double,double,double> GraphType;

bool checkMask(Mat& mask)
{
    int fgdPxls=0, bgdPxls=0;
    Point p;
    for(p.y=0; p.y < mask.rows; p.y++)
    {
        for(p.x=0; p.x < mask.cols; p.x++)
        {
            if(mask.at<uchar>(p) == GC_BGD)
            {
                bgdPxls++;
            }
            else if(mask.at<uchar>(p) == GC_FGD)
            {
                fgdPxls++;
            }
        }
    }
    return fgdPxls>=10 && bgdPxls>=10;
}

void initlearningGMMs(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM) {
    const int kMeansItCount = 10;
    const int kMeansType = KMEANS_PP_CENTERS;
//	const int KmeansType = KMEANS_RANDOM_CENTERS;

    Mat bgdLabels, fgdLabels;
    Mat bgdCenters, fgdCenters;
    vector<Vec3f> bgdSamples, fgdSamples;
    Point p;
    for (p.y = 0; p.y < img.rows; p.y++) {
        for (p.x = 0; p.x < img.cols; p.x++) {
                //GC_BGD
            if (mask.at<uchar>(p) == GC_BGD)
                bgdSamples.push_back((Vec3f) img.at<Vec3b>(p));
            else if (mask.at<uchar>(p) == GC_FGD)
                // GC_FGD
                fgdSamples.push_back((Vec3f) img.at<Vec3b>(p));
        }
    }
    CV_Assert(!bgdSamples.empty() && !fgdSamples.empty());

    Mat _bgdSamples((int) bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
    kmeans(_bgdSamples, GMM::componentsCount, bgdLabels,
            TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType,
            bgdCenters);
    Mat _fgdSamples((int) fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
    kmeans(_fgdSamples, GMM::componentsCount, fgdLabels,
            TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType,
            fgdCenters);

    bgdGMM.initLearning();
    for (int i = 0; i < (int) bgdSamples.size(); i++)
        bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
    bgdGMM.endLearning();

    fgdGMM.initLearning();
    for (int i = 0; i < (int) fgdSamples.size(); i++)
        fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
    fgdGMM.endLearning();
}

double calcBeta(const Mat& img) {
    double beta = 0;
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            Vec3d color = img.at<Vec3b>(y, x);
            if (x > 0) // left
                    {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y, x - 1);
                beta += diff.dot(diff);
            }
            if (y > 0 && x > 0) // upleft
                    {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x - 1);
                beta += diff.dot(diff);
            }
            if (y > 0) // up
                    {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x);
                beta += diff.dot(diff);
            }
            if (y > 0 && x < img.cols - 1) // upright
                    {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x + 1);
                beta += diff.dot(diff);
            }
        }
    }
    if (beta <= std::numeric_limits<double>::epsilon())
        beta = 0;
    else
        beta = 1.f
                / (2 * beta
                        / (4 * img.cols * img.rows - 3 * img.cols - 3 * img.rows
                                + 2));

    return beta;
}

void calcNWeights(const Mat& img, Mat& leftW, Mat& upleftW, Mat& upW,
        Mat& uprightW, double beta, double gamma) {
    const double gammaDivSqrt2 = gamma / std::sqrt(2.0f);
    leftW.create(img.rows, img.cols, CV_64FC1);
    upleftW.create(img.rows, img.cols, CV_64FC1);
    upW.create(img.rows, img.cols, CV_64FC1);
    uprightW.create(img.rows, img.cols, CV_64FC1);
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            Vec3d color = img.at<Vec3b>(y, x);
            if (x - 1 >= 0) // left
                    {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y, x - 1);
                leftW.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
            } else
                leftW.at<double>(y, x) = 0;
            if (x - 1 >= 0 && y - 1 >= 0) // upleft
                    {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x - 1);
                upleftW.at<double>(y, x) = gammaDivSqrt2
                        * exp(-beta * diff.dot(diff));
            } else
                upleftW.at<double>(y, x) = 0;
            if (y - 1 >= 0) // up
                    {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x);
                upW.at<double>(y, x) = gamma * exp(-beta * diff.dot(diff));
            } else
                upW.at<double>(y, x) = 0;
            if (x + 1 < img.cols - 1 && y - 1 >= 0) // upright
                    {
                Vec3d diff = color - (Vec3d) img.at<Vec3b>(y - 1, x + 1);
                uprightW.at<double>(y, x) = gammaDivSqrt2
                        * exp(-beta * diff.dot(diff));
            } else
                uprightW.at<double>(y, x) = 0;
        }
    }
}

void constructGCGraph(const Mat& img, const Mat& mask, const GMM& bgdGMM,
        const GMM& fgdGMM, double lambda, const Mat& leftW, const Mat& upleftW,
        const Mat& upW, const Mat& uprightW, GraphType* graph) {
    Point p;
    int vtxIdx=0;
    for (p.y = 0; p.y < img.rows; p.y++) {
        for (p.x = 0; p.x < img.cols; p.x++) {
            // add node
            Vec3b color = img.at<Vec3b>(p);

            // set t-weights
            double fromSource, toSink;
            if (mask.at<uchar>(p) == GC_BGD) {  // GC_BGD
                fromSource = 0;
                toSink = lambda;
            } else if (mask.at<uchar>(p) == GC_FGD) // GC_FGD
            {
                fromSource = lambda;
                toSink = 0;
            } else
            {
                fromSource = -log(bgdGMM(color));
                toSink = -log(fgdGMM(color));
            }
            graph->add_node();
            graph->add_tweights(vtxIdx, fromSource, toSink);

            // set n-weights
            if (p.x > 0) {
                double w = leftW.at<double>(p);
                graph->add_edge(vtxIdx, vtxIdx - 1, w, w);
            }
            if (p.x > 0 && p.y > 0) {
                double w = upleftW.at<double>(p);
                graph->add_edge(vtxIdx, vtxIdx - img.cols - 1, w, w);
            }
            if (p.y > 0) {
                double w = upW.at<double>(p);
                graph->add_edge(vtxIdx, vtxIdx - img.cols, w, w);
            }
            if (p.x < img.cols - 1 && p.y > 0) {
                double w = uprightW.at<double>(p);
                graph->add_edge(vtxIdx, vtxIdx - img.cols + 1, w, w);
            }
            vtxIdx++;
        }
    }
}

void estimateSegmentation(GraphType* graph, Mat& mask) {
    graph->maxflow();
    Point p;
    for (p.y = 0; p.y < mask.rows; p.y++) {
        for (p.x = 0; p.x < mask.cols; p.x++) {
            if (mask.at<uchar>(p) != GC_BGD
                    && mask.at<uchar>(p) != GC_FGD) {
                if (graph->what_segment(p.y * mask.cols + p.x) == GraphType::SOURCE)
                    mask.at<uchar>(p) = GC_PR_FGD;
                else
                    mask.at<uchar>(p) = GC_PR_BGD;
            }
        }
    }
}


void lazySnapping(const Mat& img, Mat& mask, Mat& bgdModel,
        Mat& fgdModel){

    if(!checkMask(mask))
    {
        return;
    }

    GMM bgdGMM(bgdModel), fgdGMM(fgdModel);

    initlearningGMMs(img, mask, bgdGMM, fgdGMM);

    const double gamma = 50;
    const double lambda = 9 * gamma;
    const double beta = calcBeta(img);

    Mat leftW, upleftW, upW, uprightW;
    calcNWeights(img, leftW, upleftW, upW, uprightW, beta, gamma);

    GraphType* graph = NULL;

    int vtxCount = img.cols * img.rows,
        edgeCount = 2 * (4 * img.cols * img.rows - 3 * (img.cols + img.rows) + 2);
    graph = new GraphType(vtxCount, edgeCount);

    constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, leftW, upleftW, upW,
                     uprightW, graph);

    estimateSegmentation(graph, mask);


    if(graph){
        delete graph;
    }

}

#endif // LAZYSNAPPING_H
