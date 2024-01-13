set -e

TARGET_SOC="rk3588"

# build
BUILD_DIR=build/build_linux_aarch64

if [ ! -d "${BUILD_DIR}" ]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake ../.. -DCMAKE_SYSTEM_NAME=Linux -DTARGET_SOC=${TARGET_SOC}
make -j4
make install
cd -

# relu版本
#cd install/rknn_yolov5_demo_Linux/ && ./rknn_yolov5_demo ./model/RK3588/yolov5s-640-640.rknn ../../720p60hz.mp4
# 使用摄像头
# cd install/rknn_yolov5_demo_Linux/ && ./rknn_yolov5_demo ./model/RK3588/yolov5s-640-640.rknn 0

