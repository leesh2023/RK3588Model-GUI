# import numpy as np
# from rknn.api import RKNN
#
# ONNX_MODEL = 'gru_model_hidden8_version2.onnx'  # 使用对齐后的模型
# RKNN_MODEL = 'gru_model_hidden8_version2.rknn'
#
# def show_outputs(outputs):
#     """
#     # outputs: np.ndarray of shape (3,2)  —— 单样本推理
#     for stage in range(outputs.shape[0]):
#     y, c = outputs[stage]
#     print(f"阶段{stage} → 屈服强度: {y:.2f} MPa, 导电率: {c:.2f} %IACS")
#     """
#     for i in range(outputs.shape[0]):
#         print(f"\n样本 {i}：")
#         for stage in range(3):
#             yield_pred = outputs[i, stage, 0]
#             cond_pred = outputs[i, stage, 1]
#             print(f"  阶段 {stage+1} → 屈服强度: {yield_pred:.2f} MPa, 导电率: {cond_pred:.2f} %IACS")
#
# if __name__ == '__main__':
#
#     # Create RKNN object
#     rknn = RKNN(verbose=True, verbose_file='mobilenet_build.log') # 显示日志信息
#
#     example_inputs = np.array([
#         [[1.75, 0.58, 0],
#          [900, 80, 80],
#          [400, 0.5, 0]],
#         [[1.65, 0.50, 0],
#          [800, 70, 70],
#          [350, 0.5, 0]],
#         [[1.85, 0.50, 0],
#          [850, 70, 70],
#          [400, 0.5, 0]],
#         [[1.85, 0.50, 0],
#          [750, 70, 70],
#          [400, 0.5, 0]]
#     ]
#         , dtype=np.float32)
#     dynamic_input = [
#         [[1, 3, 3]],
#         [[4, 3, 3]],
#     ]
#
#     # 变成 (1,3,3)
#     # example_inputs = example_inputs.reshape(1, 3, 3)
#
#     # Pre-process config
#     print('--> Config model')
#     # rknn.config(mean_values=[123.675, 116.28, 103.53], std_values=[58.395, 57.12, 57.375], target_platform='rk3588')
#     rknn.config(target_platform='rk3588', enable_rnn_loop=True, dynamic_input=dynamic_input)
#     print('done')
#
#     # Load model
#     print('--> Loading model')
#     ret = rknn.load_onnx(model=ONNX_MODEL)
#     if ret != 0:
#         print('Load model failed!')
#         exit(ret)
#     print('done')
#
#     # Build model
#     print('--> Building model')
#     ret = rknn.build(do_quantization=False)
#     if ret != 0:
#         print('Build model failed!')
#         exit(ret)
#     print('done')
#
#     # Export rknn model
#     print('--> Export rknn model')
#     ret = rknn.export_rknn(export_path=RKNN_MODEL, gen_cpp_demo=True) # 在导出模型的同时生成C++部署示例
#     if ret != 0:
#         print('Export rknn model failed!')
#         exit(ret)
#     print('done')
#
#     # Init runtime environment
#     print('--> Init runtime environment')
#     ret = rknn.init_runtime(target='rk3588')
#     if ret != 0:
#         print('Init runtime environment failed!')
#         exit(ret)
#     print('done')
#
#     # Inference
#     print('--> Running model')
#     outputs = rknn.inference(inputs=[example_inputs], data_format='UNDEFINED')
#     # outputs[0].shape 会是 (4,3,2)，表示 batch=4, 3 阶段, 2 指标
#     #batch_out = outputs[0]
#     single_out = outputs[0]
#     show_outputs(single_out)
#     print('done')
#
#     rknn.release()

# import numpy as np
# from rknn.api import RKNN
#
# ONNX_MODEL = 'gru_model_hidden8_version2.onnx'  # 使用对齐后的模型
# RKNN_MODEL = 'gru_model_hidden8_version2.rknn'
#
#
# def show_outputs(outputs):
#     for i in range(outputs.shape[0]):
#         print(f"\n样本 {i}：")
#         for stage in range(3):
#             yield_pred = outputs[i, stage, 0]
#             cond_pred = outputs[i, stage, 1]
#             print(f"  阶段 {stage + 1} → 屈服强度: {yield_pred:.2f} MPa, 导电率: {cond_pred:.2f} %IACS")
#
#
# if __name__ == '__main__':
#     rknn = RKNN(verbose=True)
#
#     print('--> Config model')
#     config_dict = {
#         'target_platform': 'rk3588',
#         'enable_rnn_loop': True,  # 启用RNN循环支持
#         'dynamic_input': [  # 动态输入范围
#             [[1, 3, 3]],  # 最小形状
#             [[4, 3, 3]]  # 最大形状
#         ],
#         'optimization_level': 3  # 最高优化级别
#     }
#     ret = rknn.config(**config_dict)
#     if ret != 0:
#         print(f'Config model failed! Error code: {ret}')
#         exit(ret)
#     print('done')
#
#     # 加载模型
#     print('--> Loading model')
#     ret = rknn.load_onnx(model=ONNX_MODEL)
#     if ret != 0:
#         print('Load model failed!')
#         exit(ret)
#     print('done')
#
#     # 构建模型
#     print('--> Building model')
#     ret = rknn.build(do_quantization=False)
#     if ret != 0:
#         print('Build model failed!')
#         exit(ret)
#     print('done')
#
#     # 导出模型
#     print('--> Export rknn model')
#     ret = rknn.export_rknn(export_path=RKNN_MODEL)
#     if ret != 0:
#         print('Export rknn model failed!')
#         exit(ret)
#     print('done')
#
#     # 初始化运行时
#     print('--> Init runtime environment')
#     ret = rknn.init_runtime(target='rk3588')
#     if ret != 0:
#         print('Init runtime environment failed!')
#         exit(ret)
#     print('done')
#
#     # 准备输入数据
#     example_inputs = np.array([
#         [[1.75, 0.58, 0], [900, 80, 80], [400, 0.5, 0]],
#         [[1.75, 0.58, 0], [800, 50, 50], [450, 0.5, 0]],
#         [[1.85, 0.50, 0], [850, 70, 70], [400, 0.5, 0]],
#         [[1.85, 0.50, 0], [750, 70, 70], [400, 0.5, 0]]
#     ], dtype=np.float32)
#
#     # Inference
#     print('--> Running model')
#     outputs = rknn.inference(inputs=[example_inputs], data_format='UNDEFINED')
#     # outputs[0].shape 会是 (4,3,2)，表示 batch=4, 3 阶段, 2 指标
#     #batch_out = outputs[0]
#     # inference 得到 outputs[0]，shape=(batch,12,2)
#     raw = outputs[0]  # np.ndarray, float16 → 转成 float32
#     raw = raw.astype(np.float32)
#
#     batch, C, feat = raw.shape  # C == 12, feat == 2
#     # 方向数 2，还原后每阶段2个值
#     # reshape 规则：12 = seq_len(3) * directions(2)
#     out = raw.reshape(batch, 3, 2, feat)
#     # 我们只关心最后一个方向？或者是两方向做平均？训练时 fc 接在 concat( fwd, bwd )
#     # 实际上训练时你是 concat 后接一个 fc → 它把 16 维降到 2 维，
#     # 所以这里直接把 reshape 到 (batch,3,2) 即可：
#     out = out.reshape(batch, 3, feat)
#     #single_out = outputs[0]
#     show_outputs(out)
#     print('done')
#
#     rknn.release()

import numpy as np
from rknn.api import RKNN

ONNX_MODEL = 'gru_model_hidden8_version3.onnx'   # 你的 ONNX 路径
RKNN_MODEL = 'gru_model_hidden8_version3.rknn'   # 导出 RKNN 模型路径


def show_outputs(outputs):
    """
    打印每个样本的每个阶段预测结果。
    outputs: shape=(1,3,2)，即一个样本，三个阶段，两个特征（屈服强度+导电率）
    """
    for i in range(outputs.shape[0]):
        # print(f"\n样本 {i}：")
        for stage in range(3):
            yield_pred = outputs[i, stage, 0]
            cond_pred = outputs[i, stage, 1]
            print(f"  阶段 {stage + 1} → 屈服强度: {yield_pred:.2f} MPa, 导电率: {cond_pred:.2f} %IACS")


if __name__ == '__main__':
    rknn = RKNN(verbose=True)

    print('--> Config model')
    config_dict = {
        'target_platform': 'rk3588',
        'enable_rnn_loop': True,  # 启用 RNN 循环支持
        'dynamic_input': [        # 输入动态 batch 范围
            [[4, 3, 3]],          # 最小 batch = 1
            [[4, 3, 3]]           # 最大 batch = 4
        ],
        'optimization_level': 3
    }
    ret = rknn.config(**config_dict)
    if ret != 0:
        print(f'Config model failed! Error code: {ret}')
        exit(ret)
    print('done')

    # 加载 ONNX 模型
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # 构建 RKNN 模型
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # 导出 .rknn 模型
    print('--> Export rknn model')
    ret = rknn.export_rknn(export_path=RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # 初始化推理环境
    print('--> Init runtime environment')
    ret = rknn.init_runtime(target='rk3588')
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')


    # 构造 4 个样本的输入数据
    example_inputs = np.array([
        [[1.75, 0.58, 0], [900, 80, 80], [400, 0.5, 0]],
        [[1.75, 0.58, 0], [900, 80, 80], [400, 1.0, 0]],
        [[4.08, 0.73, 0], [900, 79, 50], [450, 1.0, 0]],
        [[3.21, 0.66, 0], [900, 79, 70], [450, 1.0, 0]]
    ], dtype=np.float32)

    print('--> Running model')
    outputs = rknn.inference(inputs=[example_inputs], data_format='UNDEFINED')
    # outputs[0].shape 会是 (4,3,2)，表示 batch=4, 3 阶段, 2 指标
    #batch_out = outputs[0]
    single_out = outputs[0]
    show_outputs(single_out)
    print('done')

    # # 单样本逐条推理
    # print('--> Running model (per-sample inference)')
    # for idx, sample in enumerate(example_inputs):
    #     inp = sample[np.newaxis, ...]  # shape = (1, 3, 3)
    #     out = rknn.inference(inputs=[inp], data_format='UNDEFINED')[0]  # shape = (1, 12, 2)
    #
    #     # 转 float32
    #     out = out.astype(np.float32)
    #
    #     # reshape 到 (1, 3, 2)
    #     out = out.reshape(1, 3, 2)
    #
    #     # 打印输出
    #     print(f"\n=== 样本 {idx} ===")
    #     show_outputs(out)

    print('done')
    rknn.release()
