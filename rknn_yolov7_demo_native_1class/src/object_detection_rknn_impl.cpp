/* Copyright (C) BDI Tech Limited - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * Written by Liangchuan Gu<li.gu@bdi.tech>, Apr 2023
*/

#include <object_detection_rknn_impl.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#define _BASETSD_H

#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"
#include "rga.h"
#include "rknn_api.h"
#include "coreNum.hpp"


namespace BDI
{

/*-------------------------------------------
                  Functions
-------------------------------------------*/

std::vector<std::string> interested_labels;

bool getFileContent(std::string fileName, std::vector<std::string> & vecOfStrs)
{
    // Open the File
    std::ifstream in(fileName.c_str());
    // Check if object is valid
    if(!in)
    {
        std::cerr << "Cannot open the File : "<<fileName<<std::endl;
        return false;
    }
    std::string str;
    // Read the next line from File untill it reaches the end.
    while (std::getline(in, str))
    {
        // Line contains string of length > 0 then save it in vector
        if(str.size() > 0)
            vecOfStrs.push_back(str);
    }
    //Close The File
    in.close();
    return true;
}


static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char* load_data(FILE* fp, size_t ofst, size_t sz)
{
  unsigned char* data;
  int            ret;

  data = NULL;

  if (NULL == fp) {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0) {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char*)malloc(sz);
  if (data == NULL) {
    printf("buffer malloc failure.\n");
    return NULL;
  }
  ret = fread(data, 1, sz, fp);
  return data;
}

static unsigned char* load_model(const char* filename, int* model_size)
{
  FILE*          fp;
  unsigned char* data;

  fp = fopen(filename, "rb");
  if (NULL == fp) {
    printf("Open file %s failed.\n", filename);
    return NULL;
  }

  fseek(fp, 0, SEEK_END);
  int size = ftell(fp);

  data = load_data(fp, 0, size);

  fclose(fp);

  *model_size = size;
  return data;
}

static int saveFloat(const char* file_name, float* output, int element_size)
{
  FILE* fp;
  fp = fopen(file_name, "w");
  for (int i = 0; i < element_size; i++) {
    fprintf(fp, "%.6f\n", output[i]);
  }
  fclose(fp);
  return 0;
}

ObjectDetectionRknnImpl::ObjectDetectionRknnImpl(const std::string& model_path, const std::string& label_path, const std::string& interested_path, float nms_threshold, float box_conf_threshold) :
        status_(0), model_path_(model_path), label_path_(label_path), interested_path_(interested_path), actual_size_(0), img_width_(0), img_height_(0), img_channel_(0), model_width_(0), model_height_(0), model_channel_(0), nms_threshold_(nms_threshold), box_conf_threshold_(box_conf_threshold), resize_buf_(nullptr)
{
    // init rga context
    memset(&src_rect_, 0, sizeof(src_rect_));
    memset(&dst_rect_, 0, sizeof(dst_rect_));
    memset(&src_, 0, sizeof(src_));
    memset(&dst_, 0, sizeof(dst_));

    printf("post process config: box_conf_threshold = %.2f, nms_threshold = %.2f\n", box_conf_threshold_, nms_threshold_);


    //init(ctx_, false);

    getFileContent(interested_path_, interested_labels);
}

ObjectDetectionRknnImpl::~ObjectDetectionRknnImpl()
{
    deinitPostProcess();

    // release
    ret_ = rknn_destroy(ctx_);

    if (model_data_) {
      free(model_data_);
    }

    if (resize_buf_) {
      free(resize_buf_);
    }
}

int ObjectDetectionRknnImpl::init(rknn_context *ctx_in, bool share_weight)
{
    /* Create the neural network */
    printf("Loading model %s...\n", model_path_.c_str());
    //int            model_data_size = 0;
    //model_data_      = load_model(model_path_.c_str(), &model_data_size);
    //ret_             = rknn_init(&ctx_, model_data_, model_data_size, 0, NULL);
    //if (ret_ < 0) {
    //  printf("rknn_init error ret_=%d\n", ret_);
    //  exit(-1);
    //}

    int model_data_size = 0;
    model_data_ = load_model(model_path_.c_str(), &model_data_size);
    int ret = 0;
    // 模型参数复用/Model parameter reuse
    if (share_weight == true)
        ret = rknn_dup_context(ctx_in, &ctx_);
    else
        ret = rknn_init(&ctx_, model_data_, model_data_size, 0, NULL);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    // 设置模型绑定的核心/Set the core of the model that needs to be bound
    rknn_core_mask core_mask;
    switch (get_core_num())
    {
    case 0:
        core_mask = RKNN_NPU_CORE_0;
        break;
    case 1:
        core_mask = RKNN_NPU_CORE_1;
        break;
    case 2:
        core_mask = RKNN_NPU_CORE_2;
        break;
    }
    ret = rknn_set_core_mask(ctx_, core_mask);
    if (ret < 0)
    {
        printf("rknn_init core error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret_ = rknn_query(ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret_ < 0) {
      printf("rknn_init error ret_=%d\n", ret_);
      return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);
  
    ret_ = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
    if (ret_ < 0) {
      printf("rknn_init error ret_=%d\n", ret_);
      return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num_.n_input, io_num_.n_output);
  
    // 设置输入参数/Set the input parameters
    input_attrs_ = (rknn_tensor_attr *)calloc(io_num_.n_input, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num_.n_input; i++)
    {
        input_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs_[i]), sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs_[i]));
    }

    // 设置输出参数/Set the output parameters
    output_attrs_ = (rknn_tensor_attr *)calloc(io_num_.n_output, sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num_.n_output; i++)
    {
        output_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(output_attrs_[i]));
    }
  
    model_channel_ = 3;
    model_width_   = 0;
    model_height_  = 0;
    if (input_attrs_[0].fmt == RKNN_TENSOR_NCHW) {
      printf("model is NCHW input fmt\n");
      model_channel_ = input_attrs_[0].dims[1];
      model_height_  = input_attrs_[0].dims[2];
      model_width_   = input_attrs_[0].dims[3];
    } else {
      printf("model is NHWC input fmt\n");
      model_height_  = input_attrs_[0].dims[1];
      model_width_   = input_attrs_[0].dims[2];
      model_channel_ = input_attrs_[0].dims[3];
    }
  
    printf("model input model_height_=%d, model_width_=%d, model_channel_=%d\n", model_height_, model_width_, model_channel_);
  
    memset(inputs_, 0, sizeof(inputs_));
    inputs_[0].index        = 0;
    inputs_[0].type         = RKNN_TENSOR_UINT8;
    inputs_[0].size         = model_width_ * model_height_ * model_channel_;
    inputs_[0].fmt          = RKNN_TENSOR_NHWC;
    inputs_[0].pass_through = 0;
  
    // Init the network with one empty image
    cv::Mat img(cv::Size(model_width_, model_height_), CV_8UC3, cv::Scalar(0,0,0));

    inputs_[0].buf = (void*)img.data;
  
    gettimeofday(&start_time_, NULL);
    rknn_inputs_set(ctx_, io_num_.n_input, inputs_);
  
    rknn_output outputs[io_num_.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num_.n_output; i++)
    {
        outputs[i].want_float = 0;
    }
  
    ret_ = rknn_run(ctx_, NULL);
    ret_ = rknn_outputs_get(ctx_, io_num_.n_output, outputs, NULL);
    gettimeofday(&stop_time_, NULL);
    printf("once run use %f ms\n", (__get_us(stop_time_) - __get_us(start_time_)) / 1000);
  
    ret_ = rknn_outputs_release(ctx_, io_num_.n_output, outputs);
    return 0;
}

rknn_context *ObjectDetectionRknnImpl::get_pctx()
{
    return &ctx_;
}

std::pair<cv::Mat, Objects> ObjectDetectionRknnImpl::infer(InputData& data) 
{
    cv::Mat& img = data.roiDet;
    std::lock_guard<std::mutex> lock(mtx);
    if (!img.data) {
        std::cout << "[ERROR] invalid image!\n";
        throw;
    }

    std::pair<cv::Mat, Objects> result;
    result.second.widthOffset = data.widthOffset;
    result.second.heightOffset = data.heightOffset;
    // Expecting image in RGB format
    img_width_  = img.cols;
    img_height_ = img.rows;
    //std::cout << "img width = " << img_width_ << ", img height = " << img_height_ << "\n";
    //std::cout << "model_width_ = " << model_width_ << ", model_height_ = " << model_height_ << "\n";

    BOX_RECT pads;
    memset(&pads, 0, sizeof(BOX_RECT));
    // You may not need resize when src resulotion equals to dst resulotion
    if (img_width_ != model_width_ || img_height_ != model_height_)
    {
        //printf("resize with RGA!\n");
        if (resize_buf_ == nullptr) {
          resize_buf_ = malloc(model_height_ * model_width_ * model_channel_);
        }
        memset(resize_buf_, 0x00, model_height_ * model_width_ * model_channel_);

        src_ = wrapbuffer_virtualaddr((void*)img.data, img_width_, img_height_, RK_FORMAT_RGB_888);
        dst_ = wrapbuffer_virtualaddr((void*)resize_buf_, model_width_, model_height_, RK_FORMAT_RGB_888);
        ret_ = imcheck(src_, dst_, src_rect_, dst_rect_);
        if (IM_STATUS_NOERROR != ret_) {
        printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret_));
        throw;
      }
      IM_STATUS STATUS = imresize(src_, dst_);

      // for debug
      //cv::Mat resize_img(cv::Size(model_width_, model_height_), CV_8UC3, resize_buf_);
      //cv::imwrite("resize_input.jpg", resize_img);

      inputs_[0].buf = resize_buf_;
    } else {
      inputs_[0].buf = (void*)img.data;
    }

    gettimeofday(&start_time_, NULL);
    rknn_inputs_set(ctx_, io_num_.n_input, inputs_);

    rknn_output outputs[io_num_.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num_.n_output; i++)
    {
        outputs[i].want_float = 0;
    }

    ret_ = rknn_run(ctx_, NULL);
    ret_ = rknn_outputs_get(ctx_, io_num_.n_output, outputs, NULL);
    gettimeofday(&stop_time_, NULL);
    //printf("NN inference use %f ms\n", (__get_us(stop_time_) - __get_us(start_time_)) / 1000);

    gettimeofday(&start_time_, NULL);
    // post process
    float scale_w = (float)model_width_ / img_width_;
    float scale_h = (float)model_height_ / img_height_;

    detect_result_group_t detect_result_group;
    std::vector<float>    out_scales;
    std::vector<int32_t>  out_zps;
    for (int i = 0; i < io_num_.n_output; ++i) {
      out_scales.push_back(output_attrs_[i].scale);
      out_zps.push_back(output_attrs_[i].zp);
    }
    post_process(label_path_, (int8_t*)outputs[0].buf, (int8_t*)outputs[1].buf, (int8_t*)outputs[2].buf, model_height_, model_width_, box_conf_threshold_, nms_threshold_, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
    //BOX_RECT pads;
    //memset(&pads, 0, sizeof(BOX_RECT));
//
    //post_process((int8_t*)outputs_[0].buf, (int8_t*)outputs_[1].buf, (int8_t*)outputs_[2].buf, model_height_, model_width_, box_conf_threshold_, nms_threshold_, pads, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
    //std::cout << "[DEBUG] post_process completed\n";
    // Draw Objects
    char text[256];
    result.second.labels.resize(detect_result_group.count);
    result.second.probabilities.resize(detect_result_group.count);
    result.second.posXs.resize(detect_result_group.count);
    result.second.posYs.resize(detect_result_group.count);
    result.second.sizeXs.resize(detect_result_group.count);
    result.second.sizeYs.resize(detect_result_group.count);
    for (int i = 0; i < detect_result_group.count; i++) {
      detect_result_t* det_result = &(detect_result_group.results[i]);
      sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
      //printf("%s @ (%d %d %d %d) %f with offset (%d, %d)\n", det_result->name, det_result->box.left, det_result->box.top,
      //       det_result->box.right, det_result->box.bottom, result.second.widthOffset, result.second.heightOffset, det_result->prop);
      int x1 = det_result->box.left;
      int y1 = det_result->box.top;
      int x2 = det_result->box.right;
      int y2 = det_result->box.bottom;
      //if (std::find(interested_labels.begin(), interested_labels.end(), det_result->name) == interested_labels.end())
      //{
      //  continue;
      //}
      rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0, 255), 3);
      putText(img, text, cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0));
      result.second.posXs[i] = x1;
      result.second.posYs[i] = y1;
      result.second.sizeXs[i] = x2-x1;
      result.second.sizeYs[i] = y2-y1;
      result.second.labels[i] = det_result->name;
      result.second.probabilities[i] = det_result->prop;
    }

    //cv::imwrite("./out.jpg", img);
    ret_ = rknn_outputs_release(ctx_, io_num_.n_output, outputs);
    gettimeofday(&stop_time_, NULL);
    //printf("Post process use %f ms\n", (__get_us(stop_time_) - __get_us(start_time_)) / 1000);
    result.first = data.origImg;
    return result;
}

} // End of namespace BDI