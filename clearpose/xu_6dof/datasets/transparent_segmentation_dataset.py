import os
from scipy.io import loadmat
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import glob


class TransparentSegmentationDataset(Dataset):
	def __init__(self, data_root="Data/", object_list="Data/model/objects.csv", dataset_name='train', transforms=None, ratio=1):
		assert dataset_name in ['train', 'val', 'test']
		self.data_root = data_root
		self.dataset_name = dataset_name
		if self.dataset_name == 'train':
			sets = {
                "set4": [1, 2, 3, 4, 5],
                "set5": [1, 2, 3, 4, 5],
                "set6": [1, 2, 3, 4, 5],
                "set7": [1, 2, 3, 4, 5],
            }
		elif self.dataset_name == 'val':
			sets = {
                "set3": [4, 8, 11],
                "set7": [1, 2]
            }
		elif self.dataset_name == 'test':
			sets = {
                "set3": [1, 3],
                "set4": [6],
                "set5": [6],
				"set6": [6],
				"set7": [6],
                "set8": [1, 2]
            }
			
		self.image_list = self.get_data_list(sets, ratio)
		self.object_list = [ln.strip().split(',') for ln in open(object_list, 'r').readlines()]

		self.object_lookup_id = {obj[1]: int(obj[0]) for obj in self.object_list}
		self.object_lookup_name = {int(obj[0]): obj[1] for obj in self.object_list}

		self.transforms = transforms
	
	def get_data_list(self, sets, ratio):
		datalist = []
		for set_idx in sets:
			for scene_idx in sets[set_idx]:
				file_list = glob.glob(os.path.join(self.data_root, set_idx, f"scene{scene_idx}", "*-color.png"))
				datalist += [(set_idx, f"scene{scene_idx}", os.path.basename(f).split("-color.png")[0]) for f in file_list]
		if ratio < 1:
			random_ind = np.random.permutation(len(datalist))
			datalist = np.array(datalist)[random_ind[:int(ratio*len(datalist))]]
		return datalist
		
	def __len__(self):
		return len(self.image_list)


	def __getitem__(self, idx):
		set_path, scene_path, intid = self.image_list[idx]

		color_path = os.path.join(self.data_root, set_path, scene_path, intid+'-color.png')
		mask_path = color_path.replace('color','label')

		color = Image.open(color_path).convert("RGB")
		mask = Image.open(mask_path)

		mask = np.array(mask)
		obj_ids = np.unique(mask)
		obj_ids = obj_ids[1:]
		
		masks = mask == obj_ids[:, None, None]

		num_objs = len(obj_ids)
		boxes = []
		for i in range(num_objs):
			pos = np.where(masks[i])
			xmin = np.min(pos[1])
			xmax = np.max(pos[1])
			ymin = np.min(pos[0])
			ymax = np.max(pos[0])
			boxes.append([xmin, ymin, xmax, ymax])
		

		boxes = torch.as_tensor(boxes, dtype=torch.float32)
		labels = torch.ones((num_objs,), dtype=torch.int64)
		masks = torch.as_tensor(masks, dtype=torch.uint8)


		image_id = torch.tensor([idx])
		area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
		iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
		
		min_area = 100
		valid_area = area>min_area
		boxes = boxes[valid_area]
		area = area[valid_area]
		labels = labels[valid_area]
		masks = masks[valid_area]
		iscrowd = iscrowd[valid_area]

		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		target["masks"] = masks
		target["image_id"] = image_id
		target["area"] = area
		target["iscrowd"] = iscrowd
		target["image_path"] = color_path

		if self.transforms is not None:
			color, target = self.transforms(color, target)

		return color, target