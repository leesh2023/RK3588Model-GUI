### 文件说明  
- `.csv`是模型训练时的损失、精确率等等
- `GRU2onnx.py`将训练好的GRU模型转变为.onnx格式，导出的文件在`mmc/mmc/OA5-GRU/onnx_models/gru_model_hidden8_version3.onnx`,主要使用这个模型文件进行rknn格式转换，因为要求隐藏层对齐为8的倍数
- `GRUonnxInference.py`用于onnx模型推理
- `rknnInfer.py`用于rknn模型推理，***这是最重要的文件，主要问题也出在这里，待后期修改***
- `GRU5.py`为模型训练文件   
