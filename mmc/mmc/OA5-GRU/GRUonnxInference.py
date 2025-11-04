import onnxruntime as ort
import numpy as np


def load_session(onnx_path: str) -> ort.InferenceSession:
    """加载 ONNX 模型并返回推理 Session。"""
    return ort.InferenceSession(onnx_path)


def predict(session: ort.InferenceSession,
            inputs: np.ndarray) -> np.ndarray:
    """
    用 ONNX Runtime 做推理。
    - session: 通过 load_session 得到的 InferenceSession。
    - inputs: np.ndarray, shape = (batch_size, 3, 3), dtype=float32。
    返回:
    - outputs: np.ndarray, shape = (batch_size, 3, 2)
    """
    # 获取模型的输入输出名称（在 export 时指定过）
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 确保传入 float32
    batch_input = inputs.astype(np.float32)

    # Run 推理
    result = session.run([output_name], {input_name: batch_input})
    return result[0]  # list 中第一个就是输出张量

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


if __name__ == "__main__":
    # —— 1. 指定你的 ONNX 模型路径 ——
    onnx_model = "onnx_models/gru_model_hidden8_version3.onnx"  # 改成你要用的 hidden_size

    # —— 2. 构造示例输入 ——
    # 这里模拟一个 batch_size=2 的情况；若单样本，把 batch_size=1 即可
    # 每个样本是 3 个阶段 × 3 个工艺特征
    example_inputs = np.array([
        [[1.75, 0.58, 0], [900, 80, 80], [400, 0.5, 0]],
        [[1.75, 0.58, 0], [900, 80, 80], [400, 1.0, 0]],
        [[4.08, 0.73, 0], [900, 79, 50], [450, 1.0, 0]],
        [[3.21, 0.66, 0], [900, 79, 70], [450, 1.0, 0]]
    ], dtype=np.float32)

    # —— 3. 加载模型 & 推理 ——
    sess = load_session(onnx_model)
    # preds = predict(sess, example_inputs)  # shape (2, 3, 2)

    for idx, sample in enumerate(example_inputs):
        inp = sample[np.newaxis, ...]  # shape = (1, 3, 3)
        out = predict(sess, inp)  # shape (2, 3, 2)

        # 转 float32
        out = out.astype(np.float32)

        # reshape 到 (1, 3, 2)
        out = out.reshape(1, 3, 2)

        # 打印输出
        print(f"\n=== 样本 {idx} ===")
        show_outputs(out)

    # —— 4. 打印结果 ——
    # # preds[:,:,0] 是屈服强度预测，preds[:,:,1] 是导电率预测
    # for i in range(preds.shape[0]):
    #     # print(f"\n样本 {i}：")
    #     for stage in range(3):
    #         yield_pred = preds[i, stage, 0]
    #         cond_pred = preds[i, stage, 1]
    #         print(f"  阶段 {stage} → 屈服强度: {yield_pred:.2f} MPa, 导电率: {cond_pred:.2f} %IACS")
