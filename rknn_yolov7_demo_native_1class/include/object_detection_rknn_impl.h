/* Copyright (C) BDI Tech Limited - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Liangchuan Gu<li.gu@bdi.tech>, Apr 2023
*/

#ifndef OBJECT_DETECTION_RKNN_IMPL_H
#define OBJECT_DETECTION_RKNN_IMPL_H

#include <object_detection.h>

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <mutex>

#define _BASETSD_H

#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"
#include "rga.h"
#include "rknn_api.h"

#define PERF_WITH_POST 1
/*-------------------------------------------
                  Functions
-------------------------------------------*/

namespace BDI {


class ObjectDetectionRknnImpl: public ObjectDetection
{
public:
    ObjectDetectionRknnImpl(const std::string& model_path, const std::string& label_path, const std::string& interested_path, float nms_threshold, float box_conf_threshold);
    ~ObjectDetectionRknnImpl();

    // From ObjectDetection, the RKNN model expects image in RGB format
    virtual int init(rknn_context *ctx_in, bool isChild);
    
    virtual rknn_context *get_pctx();
    
    virtual std::pair<cv::Mat, Objects> infer(InputData& data);


private:
  int            status_;
  std::string    model_path_;
  std::string    label_path_;
  std::string    interested_path_;
  size_t         actual_size_;
  int            img_width_;
  int            img_height_;
  int            img_channel_;
  int            model_width_;
  int            model_height_;
  int            model_channel_;
  float          nms_threshold_;
  float          box_conf_threshold_;
  struct timeval start_time_;
  struct timeval stop_time_;
  int            ret_;
  std::mutex     mtx;

  unsigned char* model_data_;
  void*          resize_buf_;

  // rga context
  rga_buffer_t   src_;
  rga_buffer_t   dst_;
  im_rect        src_rect_;
  im_rect        dst_rect_;

  // rknn
  rknn_context   ctx_;
  rknn_input_output_num io_num_;
  rknn_tensor_attr *input_attrs_;
  rknn_tensor_attr *output_attrs_;
  rknn_input     inputs_[1];

};

} // End of namespace BDI

#endif //End of include guard OBJECT_DETECTION_RKNN_IMPL_H