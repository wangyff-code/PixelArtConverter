import os
import json
import base64
import io
import time
import numpy as np
import onnxruntime as ort
import webview
from PIL import Image
import sys # 确保导入 sys
# === 全局配置 ===
MODELS = {
    "128": "onnx_models/pixelart_128.onnx",
    "256": "onnx_models/pixelart_256.onnx"
}
def resource_path(relative_path):
    """ 获取资源的绝对路径，适配 PyInstaller 打包后的临时目录 """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# 确保模型目录存在
if not os.path.exists("onnx_models"):
    os.makedirs("onnx_models")

class Api:
    def __init__(self):
        self.sessions = {}

    def _get_session(self, model_key):
        # 2. 修改这里：用 resource_path 包裹模型路径
        rel_path = MODELS.get(model_key)
        if not rel_path:
            raise FileNotFoundError(f"Model key not found: {model_key}")
            
        model_path = resource_path(rel_path) # <--- 关键修改
            
        if model_key not in self.sessions:
            print(f"Loading model: {model_path}...")
            # 优先尝试 CUDA，没有则使用 CPU
            providers = ['CPUExecutionProvider']
            self.sessions[model_key] = ort.InferenceSession(model_path, providers=providers)
        
        return self.sessions[model_key]
# === 新增：保存图片的方法 ===
    def save_image(self, data_url):
        try:
            # 1. 呼出系统保存对话框
            # webview.windows[0] 获取当前主窗口
            file_path = webview.windows[0].create_file_dialog(
                webview.SAVE_DIALOG, 
                directory='', 
                save_filename='pixel_art.png',
                file_types=('PNG Image (*.png)', 'All files (*.*)')
            )

            # 如果用户点了取消，file_path 会是 None
            if not file_path:
                return {"status": "cancelled"}

            #create_file_dialog 在 save 模式下返回的是字符串路径（部分版本可能是列表，做个兼容）
            if isinstance(file_path, (list, tuple)):
                if len(file_path) > 0:
                    save_path = file_path[0]
                else:
                    return {"status": "cancelled"}
            else:
                save_path = file_path

            # 2. 解析 Base64 数据
            # data_url 格式: "data:image/png;base64,iVBORw0KGgo..."
            if ',' in data_url:
                header, encoded = data_url.split(",", 1)
            else:
                encoded = data_url
            
            img_bytes = base64.b64decode(encoded)

            # 3. 写入文件
            with open(save_path, 'wb') as f:
                f.write(img_bytes)

            return {"status": "success", "path": save_path}

        except Exception as e:
            return {"status": "error", "message": str(e)}
    def process_image(self, data):
        start_time = time.time()
        try:
            # 1. 解析参数
            img_b64 = data.get('image').split(',')[1]
            resolution = data.get('resolution', '128')
            
            # 温度限制范围 0.0001 到 1
            temperature = float(data.get('temperature', 0.01))
            temperature = max(0.0001, min(1.0, temperature)) 
            
            seed = int(data.get('seed', -1))
            noise_amp = float(data.get('noise_amp', 0.1))
            
            # 2. PIL 读取图片
            img_bytes = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_bytes))

            # 处理透明通道 (Alpha -> White Background)
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img = img.convert('RGBA')
                bg = Image.new('RGBA', img.size, (255, 255, 255, 255)) # 白底
                combined = Image.alpha_composite(bg, img)
                img = combined.convert('RGB')
            else:
                img = img.convert('RGB')

            # 3. 预处理 (Pre-processing)
            w, h = img.size
            INPUT_SIZE = 1024
            
            # 计算缩放比例，保持长宽比
            scale = INPUT_SIZE / max(h, w)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            
            # Resize
            img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # 创建 1024x1024 白色画布并居中
            canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (255, 255, 255))
            pad_left = (INPUT_SIZE - new_w) // 2
            pad_top = (INPUT_SIZE - new_h) // 2
            canvas.paste(img_resized, (pad_left, pad_top))
            
            # 转换为 Numpy 数组并归一化 (-1 到 1)
            img_np = np.array(canvas).astype(np.float32)
            img_np = (img_np / 255.0) * 2.0 - 1.0
            # [H, W, C] -> [1, C, H, W]
            img_input = img_np.transpose(2, 0, 1)[None, :, :, :]

            # 4. 推理 (Inference)
            sess = self._get_session(resolution)
            
            # 准备随机噪声
            rng = np.random.default_rng(seed if seed != -1 else None)
            noise_input = rng.standard_normal((1, 32, 256)).astype(np.float32) * noise_amp
            temp_input = np.array([temperature], dtype=np.float32)
            
            input_feed = {
                'input_image': img_input,
                'input_noise': noise_input,
                'temperature': temp_input
            }
            
            # outputs: [image, palette, attn_map]
            outputs = sess.run(None, input_feed)
            
            pred_img = outputs[0][0]     # [3, H, W]
            pred_palette = outputs[1][0] # [32, 3] <-- 直接使用模型预测的32色色卡
            
            # 5. 后处理 (Post-processing)
            pred_img = pred_img.transpose(1, 2, 0)
            pred_img = np.clip(pred_img, 0.0, 1.0) * 255.0
            pred_img = pred_img.astype(np.uint8)
            
            # 6. 裁剪逻辑 (Crop logic)
            out_h_total, out_w_total = pred_img.shape[:2]
            ratio = out_h_total / float(INPUT_SIZE)
            
            crop_top = int(pad_top * ratio)
            crop_left = int(pad_left * ratio)
            crop_h = int(new_h * ratio)
            crop_w = int(new_w * ratio)
            
            crop_h = max(1, crop_h)
            crop_w = max(1, crop_w)
            
            final_img_np = pred_img[crop_top : crop_top+crop_h, crop_left : crop_left+crop_w]
            final_pil = Image.fromarray(final_img_np)

            # 7. 处理色卡 (Palette Processing)
            # 使用 pred_palette (32, 3) 浮点数 [0,1]
            # 计算亮度: 0.299R + 0.587G + 0.114B
            lum = 0.299 * pred_palette[:, 0] + 0.587 * pred_palette[:, 1] + 0.114 * pred_palette[:, 2]
            # 获取排序索引
            indices = np.argsort(lum)
            sorted_colors = pred_palette[indices]
            
            hex_colors = []
            for color in sorted_colors:
                # 转换回 0-255 并转为 HEX
                c_val = np.clip(color * 255, 0, 255).astype(int)
                hex_val = "#{:02x}{:02x}{:02x}".format(c_val[0], c_val[1], c_val[2])
                hex_colors.append(hex_val)

            # 8. 编码图片返回
            buff = io.BytesIO()
            final_pil.save(buff, format="PNG")
            img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
            
            end_time = time.time()
            
            return {
                "status": "success", 
                "image": f"data:image/png;base64,{img_str}",
                "colors": hex_colors, # 这里返回模型预测的32色
                "time": end_time - start_time
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

if __name__ == '__main__':
    api = Api()
    webview.create_window(
        '像素画转换器 (Pixel Art Converter)', 
        'index.html', 
        js_api=api,
        width=1280, 
        height=800,
        background_color='#f0f2f5'
    )
    webview.start(debug=False)