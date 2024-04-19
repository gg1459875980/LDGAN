import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd

def process_and_save_slices(nii_path, output_dir, max_slices=160):
    print(f"Processing file: {nii_path}")
    # 加载 NIfTI 文件
    img = nib.load(nii_path)
    data = img.get_fdata()
    
    # 确保输出目录存在
    parts = nii_path.split(os.sep)
    subject_id = parts[-4]  # 假设 subject_id 在倒数第四层
    session_id = parts[-3]  # 假设 session_id 在倒数第三层
    output_path = os.path.join(output_dir, subject_id, session_id)
    os.makedirs(output_path, exist_ok=True)
    
    # 处理每个切片
    num_slices = min(data.shape[2], max_slices)
    for i in range(num_slices):
        slice = data[:, :, i]
        # 标准化切片到 0-1
        slice_normalized = (slice - np.min(slice)) / (np.max(slice) - np.min(slice) + 1e-10)
        # 保存为图像
        plt.imsave(os.path.join(output_path, f'slice_{i:03d}.png'), slice_normalized, cmap='gray')

def traverse_and_process(input_dir, output_dir):
    found_files = False
    for subdir, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.nii.gz'):
                found_files = True
                nii_path = os.path.join(subdir, file)
                process_and_save_slices(nii_path, output_dir)
    if not found_files:
        print("No .nii.gz files found in the directory.")



# 使用示例
input_dir = '/home/kevin/ABCD/testdata'  # 修改为你的输入文件夹路径
output_dir = 'data'  # 修改为你想要保存处理后数据的路径
#traverse_and_process(input_dir, output_dir)

def capitalize_after_underscore(s):
    import re
    # 使用正则表达式将下划线后的首字母变为大写
    return re.sub(r'_(\w)', lambda x: x.group(1).upper(), s)

def create_filtered_csv(input_csv_path, output_csv_path):
    # 读取原始 CSV 文件
    df = pd.read_csv(input_csv_path)

    # 处理 'src_subject_id' 和 'eventname' 列
    # 先将下划线后的首字母变为大写，然后移除所有下划线
    df['src_subject_id'] = df['src_subject_id'].apply(capitalize_after_underscore).str.replace('_', '')
    df['eventname'] = df['eventname'].apply(capitalize_after_underscore).str.replace('_', '')

    # 选择需要的列
    filtered_df = df[['src_subject_id', 'eventname', 'pps_y_ss_severity_score']]
    
    # 重命名列以符合你的需要
    filtered_df.rename(columns={
        'src_subject_id': 'subject_id',
        'eventname': 'session_id',
        'pps_y_ss_severity_score': 'clinical_score'
    }, inplace=True)

    # 构建新的 CSV 文件中的路径列
    filtered_df['session_path'] = 'data/sub-' + filtered_df['subject_id'] + '/ses-' + filtered_df['session_id']

    # 选择你需要的列
    final_df = filtered_df[['session_path', 'clinical_score']]

    # 保存为新的 CSV 文件
    final_df.to_csv(output_csv_path, index=False)

# 调用函数
input_csv_path = '/home/kevin/ABCD/core/mental-health/mh_y_pps.csv'  # 原始 CSV 文件路径
output_csv_path = 'dataset/clinic_score.csv'  # 输出 CSV 文件路径
create_filtered_csv(input_csv_path, output_csv_path)




