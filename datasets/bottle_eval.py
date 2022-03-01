from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from datasets.data_io import *
from io import BytesIO
import ipdb

class MVSDataset(Dataset):
    def __init__(self, metapath, datapath, listfile, mode, nviews, ndepths=384, interval_scale=0.01, depth_min=1.5, **kwargs):
        super(MVSDataset, self).__init__()
        self.metapath = metapath
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.depth_min = depth_min

        assert self.mode == "test"
        self.metas = self.build_list()
       

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            
            pair_file = "pair_{}.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.metapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
               
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()]
                    metas.append((scan, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)


    def read_pose_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        pose = np.fromstring(' '.join(lines[0:4]), dtype=np.float32, sep=' ').reshape((4, 4))
        extrinsics = np.linalg.inv(pose)
        intrinsics = np.array([875.000000, 0., 400.000000, 
                                    0., 875.000000, 400.000000,
                                    0., 0., 1.], dtype=np.float32).reshape((3,3))
        intrinsics[:2, :]/=4
        return intrinsics, extrinsics

    def read_bottle_img(self, filename):
        img_png = Image.open(filename) # bottles: png
        # scale 0~255 to 0~1
     
        img_jpg = img_png.convert('RGB')
        np_img = np.array(img_jpg, dtype=np.float32) / 255.
        
        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def __getitem__(self, idx):
        '''
        meta: [(tuple)]
        meta = [('1_val',26, [25,24]),('1_val', 28, [29,30]), ('1_val', 73, [68,70])]
        '''
        meta = self.metas[idx]
       
        scan, ref_view, src_views = meta
       
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []
        intrinsics_list = []
        extrinsics_list = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, 'rgb/{}_{:0>4}.png'.format(scan,vid))
            pose_filename = os.path.join(self.datapath, 'pose/{}_{:0>4}.txt'.format(scan,vid))
            # intrinsic_file =os.path.join(self.datapath, 'intrinsics.txt') 

            imgs.append(self.read_bottle_img(img_filename))
            intrinsics, extrinsics = self.read_pose_file(pose_filename)
            

            intrinsics_list.append(intrinsics)
            extrinsics_list.append(extrinsics)
            print(intrinsics, extrinsics)

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                depth_min = self.depth_min
                depth_interval = self.interval_scale
                depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval,
                                         dtype=np.float32)


        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)
        intrinsics =np.stack(intrinsics_list) 
        extrinsics =np.stack(extrinsics_list)


        return {"imgs": imgs,
                "proj_matrices": proj_matrices,
                "intrinsics":intrinsics, 
                "extrinsics":extrinsics, 
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"
                }



if __name__ == "__main__":
    dataset = MVSDataset('metadata/bottles', "/path/to/your/dataset/bottles", 'metadata/bottles/test.txt', 'test', 3,
                         128)
    item = dataset[50]
    for key, value in item.items():
        print(key, type(value))
