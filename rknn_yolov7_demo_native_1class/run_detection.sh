thread_num="$1"
echo "thread# $thread_num"

model="$2"
echo "model $model"


if [ "$model" = "7" ]; then
    echo "yolov7"
    LD_LIBRARY_PATH=./install/rknn_yolov7_demo_Linux/lib ./install/rknn_yolov7_demo_Linux/rknn_yolov7_demo ./install/rknn_yolov7_demo_Linux/model/RK3588/yolov7_tk2_RK3588_i8.rknn ./roads_720p.mp4 "$thread_num"
fi

if [ "$model" = "7t" ]; then
    echo "yolov7 tiny"
    LD_LIBRARY_PATH=./install/rknn_yolov7_demo_Linux/lib ./install/rknn_yolov7_demo_Linux/rknn_yolov7_demo ./install/rknn_yolov7_demo_Linux/model/RK3588/yolov7-tiny_tk2_RK3588_i8.rknn ./roads_720p.mp4 "$thread_num"
fi

if [ "$model" = "7b" ]; then
    echo "yolov7 tiny"
    LD_LIBRARY_PATH=./install/rknn_yolov7_demo_Linux/lib ./install/rknn_yolov7_demo_Linux/rknn_yolov7_demo ./install/rknn_yolov7_demo_Linux/model/RK3588/best_yolov7_tk2_RK3588_i8.rknn ./roads_720p.mp4 "$thread_num"
fi


if [ "$model" = "7tt" ]; then
    echo "yolov7 tiny"
    LD_LIBRARY_PATH=./install/rknn_yolov7_demo_Linux/lib ./install/rknn_yolov7_demo_Linux/rknn_yolov7_demo ./install/rknn_yolov7_demo_Linux/model/RK3588/yolov7_tiny_target.rknn ./test_target.mp4 "$thread_num"
fi
