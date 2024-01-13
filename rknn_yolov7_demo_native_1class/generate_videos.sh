thread_num="$1"
echo "thread# $thread_num"

model="$2"
echo "model $model"


if [ "$model" = "7" ]; then
    echo "yolov7 for videos/10.avi"
    LD_LIBRARY_PATH=./install/rknn_yolov7_demo_Linux/lib ./install/rknn_yolov7_demo_Linux/rknn_yolov7_demo ./install/rknn_yolov7_demo_Linux/model/RK3588/yolov7_tk2_RK3588_i8.rknn ./videos/10.avi "$thread_num"
    echo "yolov7 for videos/13.mp4"
    LD_LIBRARY_PATH=./install/rknn_yolov7_demo_Linux/lib ./install/rknn_yolov7_demo_Linux/rknn_yolov7_demo ./install/rknn_yolov7_demo_Linux/model/RK3588/yolov7_tk2_RK3588_i8.rknn ./videos/13.mp4 "$thread_num"
    echo "yolov7 for videos/14.mp4"
    LD_LIBRARY_PATH=./install/rknn_yolov7_demo_Linux/lib ./install/rknn_yolov7_demo_Linux/rknn_yolov7_demo ./install/rknn_yolov7_demo_Linux/model/RK3588/yolov7_tk2_RK3588_i8.rknn ./videos/14.mp4 "$thread_num"
    echo "yolov7 for videos/15.avi"
    LD_LIBRARY_PATH=./install/rknn_yolov7_demo_Linux/lib ./install/rknn_yolov7_demo_Linux/rknn_yolov7_demo ./install/rknn_yolov7_demo_Linux/model/RK3588/yolov7_tk2_RK3588_i8.rknn ./videos/15.avi "$thread_num"
    echo "yolov7 for videos/16.mov"
    LD_LIBRARY_PATH=./install/rknn_yolov7_demo_Linux/lib ./install/rknn_yolov7_demo_Linux/rknn_yolov7_demo ./install/rknn_yolov7_demo_Linux/model/RK3588/yolov7_tk2_RK3588_i8.rknn ./videos/16.mov "$thread_num"
fi

if [ "$model" = "7t" ]; then
    echo "yolov7 tiny for videos/10.avi"
    LD_LIBRARY_PATH=./install/rknn_yolov7_demo_Linux/lib ./install/rknn_yolov7_demo_Linux/rknn_yolov7_demo ./install/rknn_yolov7_demo_Linux/model/RK3588/yolov7-tiny_tk2_RK3588_i8.rknn ./videos/10.avi "$thread_num"
    echo "yolov7 tiny for videos/13.mp4"
    LD_LIBRARY_PATH=./install/rknn_yolov7_demo_Linux/lib ./install/rknn_yolov7_demo_Linux/rknn_yolov7_demo ./install/rknn_yolov7_demo_Linux/model/RK3588/yolov7-tiny_tk2_RK3588_i8.rknn ./videos/13.mp4 "$thread_num"
    echo "yolov7 tiny for videos/14.mp4"
    LD_LIBRARY_PATH=./install/rknn_yolov7_demo_Linux/lib ./install/rknn_yolov7_demo_Linux/rknn_yolov7_demo ./install/rknn_yolov7_demo_Linux/model/RK3588/yolov7-tiny_tk2_RK3588_i8.rknn ./videos/14.mp4 "$thread_num"
    echo "yolov7 tiny for videos/15.avi"
    LD_LIBRARY_PATH=./install/rknn_yolov7_demo_Linux/lib ./install/rknn_yolov7_demo_Linux/rknn_yolov7_demo ./install/rknn_yolov7_demo_Linux/model/RK3588/yolov7-tiny_tk2_RK3588_i8.rknn ./videos/15.avi "$thread_num"
    echo "yolov7 tiny for videos/16.mov"
    LD_LIBRARY_PATH=./install/rknn_yolov7_demo_Linux/lib ./install/rknn_yolov7_demo_Linux/rknn_yolov7_demo ./install/rknn_yolov7_demo_Linux/model/RK3588/yolov7-tiny_tk2_RK3588_i8.rknn ./videos/16.mov "$thread_num"
fi
