


# import os 

# data_dir = "/data/userdisk1/zyy/new_data/2024_PolyP_Seg/dataset/BraTS_2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/"
# # data_dir = "./data/raw_data/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/"

# all_cases = os.listdir(data_dir)

# for case_name in all_cases:
#     case_dir = os.path.join(data_dir, case_name)

#     for data_name in os.listdir(case_dir):

#         if "-" not in data_name:
#             continue
#         new_name = data_name.split("-")[-1]

#         new_path = os.path.join(case_dir, new_name)

#         old_path = os.path.join(case_dir, data_name)

#         os.rename(old_path, new_path)

#         print(f"{new_path} 命名成功")

# import os
# import shutil

# # 设置路径
# images_dir = '/media/ly/baga_seg/Low Field Images'  # 替换为你的图像目录
# labels_dir = '/media/ly/baga_seg/Subtask 2b - Basal Ganglia Segmentations'  # 替换为你的标签目录
# output_root = '/media/ly/baga_seg'  # 最终输出的目录

# # 创建输出目录（如果不存在）
# os.makedirs(output_root, exist_ok=True)

# # 遍历图像目录
# for filename in os.listdir(images_dir):
#     if filename.endswith('_ciso.nii.gz'):
#         # 提取编号，例如 LISA_0001
#         base_name = filename.replace('_ciso.nii.gz', '')

#         # 构造对应的标签文件名
#         label_filename = f'{base_name}_HF_baga.nii.gz'

#         # 构造图像和标签的完整路径
#         image_path = os.path.join(images_dir, filename)
#         label_path = os.path.join(labels_dir, label_filename)

#         # 创建以编号命名的新文件夹
#         case_folder = os.path.join(output_root, base_name)
#         os.makedirs(case_folder, exist_ok=True)

#         # 拷贝并重命名图像
#         new_image_path = os.path.join(case_folder, 'LISA_ciso.nii.gz')
#         shutil.copyfile(image_path, new_image_path)

#         # 拷贝并重命名标签
#         new_label_path = os.path.join(case_folder, 'LISA_HF_baga.nii.gz')
#         if os.path.exists(label_path):
#             shutil.copyfile(label_path, new_label_path)
#         else:
#             print(f"Warning: 标签文件不存在: {label_path}")


# import os
# import nibabel as nib
# import numpy as np
# from tqdm import tqdm

# # 标签映射规则
# label_mapping = {
#     0: 0,
#     5: 1,
#     6: 2,
#     7: 3,
#     8: 4
# }

# # 数据集根目录（里面有多个 LISA_{编号} 文件夹）
# dataset_root = "/media/ly/LISA_Task2b/baga_seg"  # <-- 请修改为你的路径

# # 遍历所有子文件夹
# for folder_name in tqdm(os.listdir(dataset_root), desc="处理标签"):
#     folder_path = os.path.join(dataset_root, folder_name)
#     if not os.path.isdir(folder_path):
#         continue

#     label_path = os.path.join(folder_path, "LISA_HF_baga.nii.gz")
#     if not os.path.exists(label_path):
#         print(f"[警告] 未找到标签文件：{label_path}")
#         continue

#     try:
#         # 读取标签文件
#         label_nii = nib.load(label_path)
#         label_data = label_nii.get_fdata().astype(np.uint8)

#         # 创建新标签图像
#         mapped_data = np.zeros_like(label_data, dtype=np.uint8)
#         for old_val, new_val in label_mapping.items():
#             mapped_data[label_data == old_val] = new_val

#         # 保存（覆盖原始标签文件）
#         new_label_nii = nib.Nifti1Image(mapped_data, affine=label_nii.affine, header=label_nii.header)
#         nib.save(new_label_nii, label_path)

#     except Exception as e:
#         print(f"[错误] 处理 {label_path} 时出错: {e}")

# print("✅ 所有标签已替换完成！")

# import os
# import shutil
# import re

# # 设置源目录和目标目录（请根据实际路径修改）
# source_dir = "/media/ly/Lisa_2025_challenge/Task 2 - Segmentation Validation"
# target_dir = "/media/ly/LISA_Task2b/baga_validation"

# # 确保目标目录存在
# os.makedirs(target_dir, exist_ok=True)

# # 遍历源目录下所有文件
# for filename in os.listdir(source_dir):
#     if filename.endswith(".nii.gz") and filename.startswith("LISA_VALIDATION_"):
#         # 提取 VALIDATION_XXXX 作为子文件夹名
#         match = re.match(r"LISA_(VALIDATION_\d{4})_ciso\.nii\.gz", filename)
#         if match:
#             folder_name = match.group(1)
#             new_filename = "LISA_ciso.nii.gz"

#             src_path = os.path.join(source_dir, filename)
#             dst_folder = os.path.join(target_dir, folder_name)
#             dst_path = os.path.join(dst_folder, new_filename)

#             # 创建子目录
#             os.makedirs(dst_folder, exist_ok=True)

#             # 复制并重命名文件
#             shutil.copy(src_path, dst_path)
#             print(f"Copied {filename} → {dst_path}")
#         else:
#             print(f"Skipped file: {filename}")


import os
import re
import nibabel as nib
import numpy as np

#—— 配置区 ——#
source_dir = "/media/ly/LISA_Task2b/results/UNETRpp"   # 原始标签文件所在目录
output_dir = "/media/ly/LISA_Task2b/submission/UNETRpp"  # 映射后文件写入目录
os.makedirs(output_dir, exist_ok=True)
#—— 结束配置 ——#

# 定义映射规则
mapping = {1: 5, 2: 6, 3: 7, 4: 8}

pattern = re.compile(r"VALIDATION_(\d{4})\.nii\.gz$")
for fname in os.listdir(source_dir):
    m = pattern.match(fname)
    if not m:
        continue

    case_id = m.group(1)  # e.g. "0001"
    src_path = os.path.join(source_dir, fname)

    # 1. 读取原始 NIfTI
    img = nib.load(src_path)
    data = img.get_fdata().astype(np.int16)

    # 2. 标签映射
    #    注意顺序：先映射高值，避免反复覆盖
    for old, new in mapping.items():
        data[data == old] = new
    # 0 保持不变

    # 3. 构造并保存新 NIfTI
    new_fname = f"LISAHF{case_id}segprediction.nii.gz"
    new_path = os.path.join(output_dir, new_fname)
    new_img = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(new_img, new_path)

    print(f"Processed {fname} → {new_fname}")

