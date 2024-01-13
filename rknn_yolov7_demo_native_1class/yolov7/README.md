## 原始 checkpoint 文件：

| 模型        | 版本  | 激活函数  | 百度盘提供 | 链接(路径)                                                   |
| ----------- | ----- | --------- | ---------- | ------------------------------------------------------------ |
| yolov7-tiny | v0.1  | leakyrelu | 否         | https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt |
| yolov7      | v.0.1 | silu      | 否         | https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt |

* 若权重文件已经在原始仓库中提供，则 airockchip 不会再单独提供一次





## 固化模型(可直接转为RKNN)：

Toolkit1:

| 模型        | 版本 | 激活函数  | 链接(路径)                                  |
| ----------- | ---- | --------- | ------------------------------------------- |
| yolov7-tiny | v6.2 | leakyrelu | ./deploy_models/toolkit1/yolov7-tiny_tk1.pt |
| yolov7      | v6.2 | silu      | ./deploy_models/toolkit1/yolov7_tk1.pt      |

对应的RKNN模型位于 [./deploy_models/toolkit1/model_cvt](./deploy_models/toolkit1/model_cvt)



Toolkit2:

| 模型        | 版本 | 激活函数  | 链接(路径)                                  |
| ----------- | ---- | --------- | ------------------------------------------- |
| yolov7-tiny | v6.2 | leakyrelu | ./deploy_models/toolkit2/yolov7-tiny_tk2.pt |
| yolov7      | v6.2 | silu      | ./deploy_models/toolkit2/yolov7_tk2.pt      |

对应的RKNN模型位于 [./deploy_models/toolkit2/model_cvt](./deploy_models/toolkit2/model_cvt)

