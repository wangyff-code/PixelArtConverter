from PIL import Image
import os

def convert_to_ico(input_path, output_path=None):
    """
    将图片转换为 ICO 格式
    :param input_path: 源图片路径 (png, jpg, jpeg等)
    :param output_path: 输出 ico 路径
    """
    try:
        # 如果没有指定输出路径，则使用原文件名，后缀改为 .ico
        if not output_path:
            output_path = os.path.splitext(input_path)[0] + ".ico"

        # 打开图片
        img = Image.open(input_path)

        # ICO 通常包含多种尺寸以适应 Windows 的不同显示模式
        # 常见尺寸：16x16, 32x32, 48x48, 64x64, 128x128, 256x256
        icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]

        # 转换并保存
        # 如果图片是 RGBA (带透明度)，Pillow 会自动处理
        img.save(output_path, format='ICO', sizes=icon_sizes)
        
        print(f"转换成功！图标已保存至: {output_path}")

    except Exception as e:
        print(f"转换失败: {e}")

if __name__ == "__main__":
    # --- 你只需要修改这里 ---
    input_image = "QQ图片20260121192322.png"  # 你的原始图片路径
    # -----------------------
    
    if os.path.exists(input_image):
        convert_to_ico(input_image)
    else:
        print(f"找不到文件: {input_image}")