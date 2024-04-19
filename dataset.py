import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob

class MRIDataset(Dataset):
    def __init__(self, data_dir, scores_csv, transform=None):
        """
        Args:
            data_dir (string): Directory with all the MRI images.
            scores_csv (string): Path to the csv file with clinical scores.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.clinic_scores = pd.read_csv(scores_csv)
        self.transform = transform
        self.sessions = self._load_sessions(data_dir)
    

    def _load_sessions(self, data_dir):
        sessions = []
        subs = sorted(glob.glob(os.path.join(data_dir, 'sub*')))
        for sub_dir in subs:
            ses_dirs = sorted(glob.glob(os.path.join(sub_dir, 'ses*')), key=self._parse_ses)
            for ses_dir in ses_dirs:
                sessions.append(ses_dir)
        return sessions

    def _parse_ses(self, ses_path):

        ses_name = os.path.basename(ses_path)  # 例如 'ses-baselineYear1Arm1'
        if 'baseline' in ses_name:
            return 0
        else:
            # 从ses_name中提取年份数字，例如 '1YearFollowUpYArm1' 中的 '1'
            year_number = int(''.join(filter(str.isdigit, ses_name.split('Year')[0])))
            return year_number

    
    def __len__(self):
        return len(self.sessions)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ses_path = self.sessions[idx]
        # Assuming the session_path in csv file is the relative path from data_dir
        relative_ses_path = os.path.relpath(ses_path, self.data_dir)
        score = self.clinic_scores.loc[self.clinic_scores['session_path'] == relative_ses_path, 'clinical_score'].values[0]
        
        # 解析sub和ses信息
        sub_ses_info = relative_ses_path.split(os.sep)[:2]  # 使用os.sep确保操作系统兼容性
        sub_info, ses_info = sub_ses_info
        
        # Load all slices for a session and stack them into a single 3D array
        slice_paths = sorted(glob.glob(os.path.join(ses_path, '*.png')))
        slices = [Image.open(slice_path) for slice_path in slice_paths]
        
        # Apply any transforms if specified
        if self.transform:
            slices = [self.transform(slice_img) for slice_img in slices]
        
        # Stack slices to form a single 3D image
        mri_image = torch.stack(slices)
        
        ses_map = {
            'ses-baselineYear1Arm1' : 0,
            'ses-1YearFollowUpYArm1' : 1,
            'ses-2YearFollowUpYArm1' : 2,
            'ses-3YearFollowUpYArm1' : 3,
            'ses-4YearFollowUpYArm1' : 4,

        }

        sample = { 
            'sub_info': sub_info,  # sub信息
            'ses_info': ses_map[ses_info],  # ses信息, 
            'image': mri_image, 
            'score': score,
            
        }

        return sample

# Define any transforms you want to apply to each slice
transform = transforms.Compose([
    transforms.ToTensor(),
    # Add any additional transforms here
])

# Create the dataset
dataset = MRIDataset(data_dir='data/', scores_csv='dataset/clinic_score.csv', transform=transform)

print(dataset[0]['image'].shape)
