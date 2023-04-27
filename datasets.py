from torch.utils.data import Dataset
import numpy as np
from torch import from_numpy
from torch.nn.functional import one_hot
from augmentations import GeomTransform

class LiverDataset(Dataset):

    def __init__(self,dataframe,ct_transform = None,aug_transform=None) -> None:
        '''
        dataframe : pandas.DataFrame
            Dataframe containing the following columns:
                patient_id : int
                    Patient ID
                slice_id : int
                    Slice ID within the patient scan
                ct_path : str
                    Path to the CT slice as an npy file
                mask_path : str
                    Path to the segmentation mask as an npy file

        transform : torch.transforms
            Transform to apply to the CT and segmentation mask

        Returns
        -------
        ct_scan : torch.Tensor
            (1,512,512) tensor containing the CT scan
        seg_scan : torch.Tensor
            (1,512,512) tensor containing the segmentation mask
        (patient_id, slice_id) : tuple
            Tuple containing the patient ID and slice ID
        '''
        self.dataframe = dataframe
        self.ct_transform = ct_transform
        assert isinstance(aug_transform, list) or aug_transform is None , 'aug_transform must be a list'
        self.aug_transform = aug_transform

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        ct_scan = from_numpy(np.load(row['ct_path'])).unsqueeze(0).float()
        seg_scan = from_numpy(np.load(row['mask_path'])).unsqueeze(0)

        if self.ct_transform:
            ct_scan = self.ct_transform(ct_scan)
        if self.aug_transform:
            for aug in self.aug_transform:
                if isinstance(aug,GeomTransform):
                    aug.update_params()
                    seg_scan = aug(seg_scan)
                ct_scan = aug(ct_scan)
        # One hot encoding of the segmentation mask
        seg_scan = one_hot(seg_scan.long(), num_classes=3).squeeze(0).float().permute(2,0,1)

        return ct_scan, seg_scan, (row['patient_id'], row['slice_id'])