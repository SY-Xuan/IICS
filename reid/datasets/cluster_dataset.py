from __future__ import print_function, absolute_import
import os.path as osp


class Cluster(object):
    def __init__(self, root, cluster_result_cam, cam_id):
        self.root = root
        self.train_set = []
        classes = []
        for fname, pid in cluster_result_cam.items():
            self.train_set.append((fname, pid, cam_id))
            classes.append(pid)
        self.classes_num = len(set(classes))

    @property
    def images_dir(self):
        return osp.join(self.root, 'images')
