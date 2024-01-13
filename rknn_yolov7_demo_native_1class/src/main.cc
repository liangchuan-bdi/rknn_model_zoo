#include <stdio.h>
#include <cstdlib>
#include <memory>
#include <sys/time.h>
#include <iostream>

//#include "opencv2/core/core.hpp"
#include <opencv2/videoio.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "rknnPool.hpp"
#include <object_detection_rknn_impl.h>

void addOffset(BDI::Objects& objects, int widthOffset, int heightOffset)
{
  auto totalDets = objects.posXs.size();
  for(auto index = 0; index < totalDets; ++index) {
    objects.posXs[index] += widthOffset;
    objects.posYs[index] += heightOffset;
  }
}

namespace {

struct Det{
  float posX_;
  float posY_;
  float sizeX_;
  float sizeY_;
  float probability_;
  std::string label_;
  Det(float posX, float posY, float sizeX, float sizeY, float probability, const std::string& label):
  posX_(posX), posY_(posY), sizeX_(sizeX), sizeY_(sizeY), probability_(probability), label_(label) {}
};

} // Anonymous namespace

void mergeDetections(cv::Mat& img, const std::vector<BDI::Objects>& detectionVec, BDI::Objects& objects)
{
  int detection_stride_width_ = 640;
  int stitch_sensitivity_ = 5;
  std::vector<Det> dets;
  int totalDetCount = 0;
  for(auto index = 0; index < detectionVec.size(); ++index)
  {
    const auto& det = detectionVec[index];
    totalDetCount += det.posXs.size();
    auto currentDetectionCount = det.posXs.size();
    for (auto detIndex = 0; detIndex < currentDetectionCount; ++detIndex)
    {
      if(det.sizeXs[detIndex] == 0 || det.sizeYs[detIndex] == 0) {
        continue;
      }
      dets.emplace_back(det.posXs[detIndex], det.posYs[detIndex],det.sizeXs[detIndex], det.sizeYs[detIndex], det.probabilities[detIndex], det.labels[detIndex]);
    }
  }
  // Sort detections by posX
  std::sort(dets.begin(), dets.end(),
      [] (const Det& d1, const Det& d2)
      {
          return (d1.posX_ < d2.posX_);
      }
  );
  // Merge adjacent bboxes
  float prevX = INT_MIN;
  float prevXPlusWidth = INT_MIN;
  std::string prevLabel = "";
  std::vector<Det> mergedDets;
  for(auto& det : dets) {
      auto detXPlusWidth = det.posX_ + det.sizeX_;
      if(std::abs(prevXPlusWidth - detection_stride_width_) > stitch_sensitivity_){
        mergedDets.push_back(det);
        prevX = det.posX_;
        prevXPlusWidth = detXPlusWidth;
        prevLabel = det.label_;
        continue;
      }
      if(det.posX_ > prevXPlusWidth + stitch_sensitivity_) {
          mergedDets.push_back(det);
          prevX = det.posX_;
          prevXPlusWidth = detXPlusWidth;
          prevLabel = det.label_;
      } else {
          // Only when they are the same class and it is close to the separation
          if(det.label_ != prevLabel) {
            prevX = det.posX_;
            prevXPlusWidth = detXPlusWidth;
            prevLabel = det.label_;
          } else 
          {
            auto last = mergedDets.back();
            mergedDets.pop_back();
            // Merge with the outer region including both bboxes
            prevX = last.posX_; // The posX
            prevXPlusWidth = detXPlusWidth + last.sizeX_;
            auto mergedY = std::min(det.posY_, last.posY_);
            mergedDets.emplace_back(prevX, mergedY, last.sizeX_ + det.sizeX_, std::max(last.posY_ + last.sizeY_, det.posY_+det.sizeY_) - mergedY, std::max(last.probability_, det.probability_), last.label_);
          }
      }
  }
  totalDetCount = mergedDets.size();
  //ROS_INFO_STREAM("Merging " << totalDetCount << " detections");
  objects.labels.resize(totalDetCount);
  objects.probabilities.resize(totalDetCount);
  objects.posXs.resize(totalDetCount);
  objects.posYs.resize(totalDetCount);
  objects.sizeXs.resize(totalDetCount);
  objects.sizeYs.resize(totalDetCount);
  //objects.type_ids.resize(totalDetCount);

  int targetIndex = 0;
  for(auto index = 0; index < totalDetCount; ++index)
  {
    const auto& det = mergedDets[index];
    objects.labels[targetIndex] = det.label_;
    objects.probabilities[targetIndex] = det.probability_;
    objects.posXs[targetIndex] = det.posX_;
    objects.posYs[targetIndex] = det.posY_;
    objects.sizeXs[targetIndex] = det.sizeX_;
    objects.sizeYs[targetIndex] = det.sizeY_;
    //objects.type_ids[targetIndex] = det.classId_;
    //ROS_INFO_STREAM("Drawing detection result: (" << objects.posXs[targetIndex] << ", " << objects.posYs[targetIndex] << ", " << objects.sizeXs[targetIndex] << ", " << objects.sizeYs[targetIndex] << "), class: " << objects.labels[targetIndex] << ", class ID: " << objects.type_ids[targetIndex]);
    auto label = objects.labels[targetIndex];
    std::transform(label.begin(), label.end(), label.begin(), [](unsigned char c){ return std::tolower(c); });
    if(label == "car")
    {
        cv::rectangle(img, cv::Point(objects.posXs[targetIndex], objects.posYs[targetIndex]), cv::Point(objects.posXs[targetIndex] + objects.sizeXs[targetIndex],  objects.posYs[targetIndex] + objects.sizeYs[targetIndex]), cv::Scalar(255, 0, 0, 255), 3);
    }
    //cv::putText(img, objects.labels[targetIndex], cv::Point(objects.posXs[targetIndex], objects.posYs[targetIndex] + 12), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0));
    ++targetIndex;
  }
}

void poolDetect(int threadNum, int frames, cv::VideoWriter& outputVideo, rknnPool<BDI::ObjectDetectionRknnImpl, BDI::InputData, std::pair<cv::Mat, BDI::Objects> >& detector, cv::Mat& img, BDI::Objects& objects)
{
  int detection_stride_width_ = 640;
  int detection_stride_height_ = 0;
  int getResult = 0;
  cv::Mat originalImg;
  if(detection_stride_width_ > 0 || detection_stride_height_ > 0) {
    std::vector<BDI::Objects> objectsVec;
    int totalDet = 0;
    int rowStrideCount = 0;
    int colStrideCount = 0;
    int widthMargin = 0;
    int heightMargin = 0;
    //std::cout << "[DEBUG] img.cols: " << img.cols << ", img.rows: " << img.rows << "\n";
    if (detection_stride_width_ > 0) {
      rowStrideCount = (int)(img.cols / detection_stride_width_);
      widthMargin = (int)((img.cols % detection_stride_width_) / 2);
      totalDet += rowStrideCount;
    }
    if (detection_stride_height_ > 0) {
      colStrideCount = (int)(img.rows / detection_stride_height_); 
      heightMargin = (int)((img.rows % detection_stride_height_)/2);
      totalDet *= colStrideCount;
    } else {
      heightMargin = (int)((img.rows % 640)/2);
    }
    
    for (auto index = 0; index < totalDet; ++index){
      int heightOffset = (int)(index / rowStrideCount) * detection_stride_height_ + heightMargin;
      int col = index - (int)(index / rowStrideCount) * rowStrideCount;
      int widthOffset = col * detection_stride_width_  + widthMargin;
      cv::Rect roi(widthOffset, heightOffset, 640, 640);

      auto imgRoi = img(roi).clone();
      //BDI::Objects detections;
      //cv::imwrite("/root/shared_volume/bdi_drone/det_" + std::to_string(heightOffset) + "_" + std::to_string(widthOffset) + ".jpg", imgRoi);
      //std::cout << "[DEBUG] adding " << index << "th detection result with widthOffset: " << widthOffset << ", heightOffset: " << heightOffset << " to " << index << "th part\n";
      BDI::InputData data(imgRoi, img.clone(), widthOffset, heightOffset);
      if (detector.put(data) != 0) {
        throw std::runtime_error("ERROR: cannot add detection task to the pool!");
      }
      std::pair<cv::Mat, BDI::Objects> detResult;
      if (frames >= threadNum && detector.get(detResult) != 0)
      {
        break;
      }
      addOffset(detResult.second, detResult.second.widthOffset, detResult.second.heightOffset);
      objectsVec.push_back(detResult.second);
      originalImg = detResult.first;
    }
    mergeDetections(originalImg, objectsVec, objects);
  } else {
    //obj_detection_ptr_->detect(img, objects);
    BDI::InputData data(img, img, 0, 0);
      if (detector.put(data) != 0) {
        throw std::runtime_error("ERROR: cannot add detection task to the pool!");
      }
    std::pair<cv::Mat, BDI::Objects> detResult;
      if (frames >= threadNum && detector.get(detResult) != 0)
      {
        return;
      }
    originalImg = detResult.first;
  }
  outputVideo << originalImg;
}

int main(int argc, char **argv)
{
    char *model_name = NULL;
    if (argc != 4)
    {
        printf("Usage: %s <rknn model> <jpg> <thread#>\n", argv[0]);
        return -1;
    }
    // 参数二，模型所在路径/The path where the model is located
    model_name = (char *)argv[1];
    // 参数三, 视频/摄像头
    char *vedio_name = argv[2];

    // 初始化rknn线程池/Initialize the rknn thread pool
    int threadNum = atoi(argv[3]);
    //rknnPool<rkYolov5s, cv::Mat, cv::Mat> testPool(model_name, threadNum, "./model/coco_80_labels_list.txt", "./model/coco_80_labels_list.txt", 0.5, 0.5);
    rknnPool<BDI::ObjectDetectionRknnImpl, BDI::InputData, std::pair<cv::Mat, BDI::Objects> > testPool(model_name, threadNum, "./model/bdi_labels_list.txt", "./model/bdi_labels_list.txt", 0.5, 0.5);
    if (testPool.init() != 0)
    {
        printf("rknnPool init fail!\n");
        return -1;
    }

    cv::namedWindow("Camera FPS");
    cv::VideoCapture capture;
    if (strlen(vedio_name) == 1)
        capture.open((int)(vedio_name[0] - '0'));
    else
        capture.open(vedio_name);

    cv::Size S = cv::Size((int) capture.get(cv::CAP_PROP_FRAME_WIDTH),    // Acquire input size
                  (int) capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter outputVideo;                                        // Open the output
    auto outputName = "./output/" + std::string(vedio_name) + std::string("_detection_result.mp4");
    int ex = static_cast<int>(capture.get(cv::CAP_PROP_FOURCC));     // Get Codec Type- Int form
    outputVideo.open(outputName, ex, capture.get(cv::CAP_PROP_FPS), S, true);
    if (!outputVideo.isOpened())
    {
        std::cout  << "Could not open the output video for write: " << outputName << std::endl;
        return -1;
    }
    struct timeval time;
    gettimeofday(&time, nullptr);
    auto startTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    int frames = 0;
    auto beforeTime = startTime;
    while (capture.isOpened())
    {
        cv::Mat img;
        if (capture.read(img) == false)
            break;
        //if (testPool.put(img) != 0)
        //    break;
//
        //std::pair<cv::Mat, BDI::Objects> detResult;
        //if (frames >= threadNum && testPool.get(detResult) != 0)
        //    break;

        BDI::Objects objects;
        poolDetect(threadNum, frames, outputVideo, testPool, img, objects);

        //cv::imshow("Camera FPS", img);
        //if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出/Press q to exit
        //    break;
        // TODO: draw detection results
        
        frames++;

        if (frames % 120 == 0)
        {
            gettimeofday(&time, nullptr);
            auto currentTime = time.tv_sec * 1000 + time.tv_usec / 1000;
            printf("120帧内平均帧率:\t %f fps/s\n", 120.0 / float(currentTime - beforeTime) * 1000.0);
            beforeTime = currentTime;
        }
    }

    // 清空rknn线程池/Clear the thread pool
    while (true)
    {
        std::pair<cv::Mat, BDI::Objects> detResult;
        if (testPool.get(detResult) != 0)
            break;
        //cv::imshow("Camera FPS", detResult.first);
        //if (cv::waitKey(1) == 'q') // 延时1毫秒,按q键退出/Press q to exit
        //    break;
        outputVideo << detResult.first;
        frames++;
    }

    gettimeofday(&time, nullptr);
    auto endTime = time.tv_sec * 1000 + time.tv_usec / 1000;

    printf("Average:\t %f fps/s\n", float(frames) / float(endTime - startTime) * 1000.0);

    return 0;
}
