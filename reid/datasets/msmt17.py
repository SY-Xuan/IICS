from __future__ import print_function, absolute_import
import os.path as osp

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json
from tqdm import tqdm


class MSMT17(Dataset):
    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(MSMT17, self).__init__(root, split_id=split_id)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_val)

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        print("Generate dataset this step may take some times")
        import re
        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        fpath = osp.join(raw_dir, 'MSMT17_V2.zip')
        if not osp.isfile(fpath):
            raise RuntimeError("Please download the dataset manually")

        # Extract the file
        exdir = osp.join(raw_dir, 'msmt17')
        if not osp.isdir(exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=raw_dir)

        # Format
        images_dir = osp.join(self.root, 'images')
        print(self.root)
        mkdir_if_missing(images_dir)
        exdir = raw_dir
        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        identities = []
        all_pids = {}

        def register(subdir, txt_file):
            with open(osp.join(exdir, txt_file), "r") as f:
                fpaths = f.readlines()
            pids = set()
            fnames = []
            for fpath in tqdm(fpaths):
                fpath = fpath.replace("\n", '')
                filename = fpath.split(" ")[0]
                pid = int(fpath.split(" ")[1])
                # deal with the query and gallery
                if subdir == "mask_test_v2":
                    pid += 1500
                fname = osp.basename(filename)
                cam = int(fname.split("_")[2])
                assert 1 <= cam <= 15
                cam -= 1
                if pid not in all_pids:
                    all_pids[pid] = len(all_pids)
                pid = all_pids[pid]
                pids.add(pid)
                if pid >= len(identities):
                    assert pid == len(identities)
                    identities.append([[]
                                       for _ in range(15)])  # 15 camera views
                fname = ('{:08d}_{:02d}_{:04d}.jpg'.format(
                    pid, cam, len(identities[pid][cam])))
                identities[pid][cam].append(fname)
                fnames.append(fname)
                shutil.copy(osp.join(exdir, subdir, filename),
                            osp.join(images_dir, fname))
            return pids, fnames

        trainval_pids, _ = register('mask_train_v2', "list_train.txt")
        gallery_pids, gallery_names = register('mask_test_v2',
                                               "list_gallery.txt")
        query_pids, query_names = register('mask_test_v2', "list_query.txt")
        assert query_pids <= gallery_pids

        # Save meta information into a json file
        meta = {
            'name': 'msmt17',
            'shot': 'multiple',
            'num_cameras': 15,
            'identities': identities,
            'gallery_names': gallery_names,
            'query_names': query_names
        }
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training / test split
        splits = [{
            'trainval': sorted(list(trainval_pids)),
            'query': sorted(list(query_pids)),
            'gallery': sorted(list(gallery_pids))
        }]
        write_json(splits, osp.join(self.root, 'splits.json'))
