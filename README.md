算法之颠：VLSOT 自动端到端跟踪

2026 VLP 挑战赛参赛作品。

团队：算法之颠  
负责人：王魏魏  
参赛赛道：专项命题挑战赛道 A（VLSOT）  
代码仓库：https://github.com/1339400321/algorithm-peak-vlsot-2026

本仓库实现赛道 A：输入单目 RGB 视频和自然语言目标描述，自动选择目标车辆，并输出三维位置轨迹、目标尺寸和置信度。

## 方法

1. 使用 torchvision Faster R-CNN COCO 权重产生车辆候选框。
2. 从描述中解析车型、颜色、距离、方位区域、可见性和朝向等信息。
3. 对候选框进行文字属性、颜色、车型和空间位置综合排序。
4. 使用动态规划保持 30 帧目标轨迹连续。
5. 使用训练集几何原型和三帧二维框估计三维位置。

测试推理不读取测试答案、不使用人工测试框、不使用测试视频编号特例。

## 运行

准备比赛数据集和 Faster R-CNN 权重后运行：

```powershell
python auto_infer.py `
  --dataset VLSOT-dataset/dataset `
  --detector-weights fasterrcnn_resnet50_fpn_coco-258fb6c6.pth `
  --geometry-model auto_geometry_model.npz `
  --output output
```

`auto_geometry_model.npz` 已包含训练得到的几何原型。比赛数据集不随仓库发布，避免违反数据集许可。

## 输出

输出目录包含 `automatic_result.zip`、`prediction.json`、`run_manifest.json` 和预览图。请以比赛平台提交页面显示的当前格式要求为准；正式平台验证的 V5 结果包单独保存在项目交付目录中。

## 验证

五折场景留出验证共 6810 帧，当前 v5 配置的本地位置误差均值约 2.62 米，误差小于 0.5 米的代理比例约 32.42%。这些是本地代理指标，不是平台官方 Recall@0.5；最终成绩以平台评测为准。

## 权重与许可证

模型权重下载地址：百度网盘 <https://pan.baidu.com/s/1Hb5KWe9dAg67CAW-lkWrpA>

提取码：`iihf`

网盘包包含 `fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`、`fasterrcnn_detector.onnx` 和 `auto_geometry_model.npz`。其中 Faster R-CNN COCO 权重来自 torchvision/COCO，需遵守对应第三方许可证；自有代码采用 MIT 许可证。权重文件不放入 GitHub，运行前请从上述链接下载并按“运行”部分的参数指定路径。
