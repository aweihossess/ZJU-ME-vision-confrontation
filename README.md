# 机器人视觉竞技A竞赛

## 代码的框架和主流程
- main.py的run()是主流程
- UpAPI提供了sensor, action, detector的封装和接口
- sensor主要是封装了如何去读取摄像头和灰度传感器
- action封装了对底盘运动的实现接口，主要涉及嵌入式的通信
- detector包含了多个检测模型的加载和调用

## 待调试优化参数
- [ ] IMU校准
- [ ] 测试extra_sleep_delay是否有必要
- [ ] 确认api.stop()是否有必要
- [ ] 测试寻找k_back_error_correction更合适的值
- [ ] rotation_90_time和rotation_180_time是否有更合适的值
- [ ] 各段的distance，还需要调整寻找最合适的值
- [ ] arm_action_duration等参数还可以调整
- [ ] target_width_face需要调整到一个更好的值
- [ ] adjust_position里4个参数还需要调整

## 学习建议
- 面向对象编程
- 计算机组成+操作系统
- Python的基础语法
- 深度学习的基本概念 CV 分割, 检测, 姿态估计, 分类
- 尝试标注、找数据、训练/微调、得到模型、推理
- 嵌入式 上位机+下位机

- top->down: 从现成的结果到原理设计
- down->top: 从原理设计到最终项目

## 操作参考
- scp -r /本地路径/ username@linux_ip:/远程路径/
```
  scp -r D:\code\robotics_vision\main.py bcsh@192.168.3.38:~/jiangchaofeng/
```