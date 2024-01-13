/* Copyright (C) BDI Tech Limited - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Liangchuan Gu<li.gu@bdi.tech>, Apr 2023
*/

#ifndef OBJECT_DETECTION_H
#define OBJECT_DETECTION_H

#include "opencv2/core/core.hpp"
#include "rknn_api.h"
#include <vector>
#include <string>

namespace BDI {

struct Objects
{
    int widthOffset;
    int heightOffset;
    std::vector<std::string> labels;
    std::vector<float> probabilities;
    std::vector<float> posXs;
    std::vector<float> posYs;
    std::vector<float> sizeXs;
    std::vector<float> sizeYs;
};

struct InputData
{
    cv::Mat roiDet;
    cv::Mat origImg;
    int widthOffset;
    int heightOffset;
    InputData(cv::Mat& roi, const cv::Mat& img, int wOffset, int hOffset) : 
     roiDet(roi), origImg(img), widthOffset(wOffset), heightOffset(hOffset)
     {}
};

class ObjectDetection
{
public:
    int init(rknn_context *ctx_in, bool isChild);
    
    rknn_context *get_pctx();
    
    std::pair<cv::Mat, Objects> infer(InputData& data);


    // enable deletion of a Derived* through a Base*
    virtual ~ObjectDetection() = default;    
}; // End of interface class ObjectDetection


} // End of namespace BDI

#endif  // End of include guard OBJECT_DETECTION_H
