# import numpy as np
# data = np.load('/media/ly/LISA_Task2b/train_fullres_process/LISA_0001.npz')
# # print(data.files)
# # for key in data.files:
# #     value = data[key]
# #     print(f"\n键名: {key}")
# #     print(f"数据类型: {value.dtype}")
# #     print(f"数据形状: {value.shape}")
# #     print(f"唯一值（若可行）: {np.unique(value)}")

# image = data['data']  # 键名可能是 'image'，视情况而定

# # print("图像数据 shape:", image.shape)
# # print("通道数 C =", image.shape[0])

# print("C维数据的取值范围：", np.min(image[0]), "到", np.max(image[0]))
# print("C维数据的唯一值（部分）:", np.unique(image[0])[:10])  # 只显示前10个唯一值

import nibabel as nib
import numpy as np

# datapath='/media/ly/LISA_Task2b/submission/LISAHF1015segprediction_fixed.nii.gz'

# data = nib.load(datapath)       
# data = data.get_fdata()
# bvals, bvecs = read_bvals_bvecs(bvalpath, bvecpath)
# gtab = gradient_table(bvals,bvecs)

# print(data.shape)
# print(bvals.shape)
# print(bvecs.shape)
# print(gtab.bvals)

# 检查 affine 是否一致, 如果不一致，可以强制使用 GT 的 affine
gt_path = "/media/ly/LISA_Task2b/baga_seg/LISA_1015/LISA_HF_baga.nii.gz"
pred_path = "/media/ly/LISA_Task2b/submission/ourvalidation/LISAHF1015segprediction.nii.gz"

pred_aff = nib.load(pred_path).affine
gt_aff = nib.load(gt_path).affine

print(np.allclose(pred_aff, gt_aff))

gt_nii = nib.load(gt_path)
pred_nii = nib.load(pred_path)
pred_fixed = nib.Nifti1Image(pred_nii.get_fdata(), affine=gt_nii.affine, header=gt_nii.header)
nib.save(pred_fixed, '/media/ly/LISA_Task2b/submission/LISA1015_pred_fixed_affine.nii.gz')