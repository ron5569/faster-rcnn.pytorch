from datasets.imdb import imdb

import os
import numpy as np
import xml.etree.ElementTree as ET
import scipy.sparse
from os.path import join

from datasets.voc_eval import voc_eval


class custom(imdb):
    def __init__(self, name, is_sort = False):
        imdb.__init__(self, name)
        self._data_path = "/media/indoordesk/653ce34c-0c14-4427-8029-be7afe6d1989/test"
        #self._data_path = "/mnt/usb-Seagate_Portable_NAA5TXW7-0:0-part2/drone_db_mixed/train/"
        self._classes = ('__background__', 'person' # always index 0
                         )
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index(is_sort)
        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]

        return gt_roidb

    def _load_image_set_index(self, is_sort):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_index = os.listdir(join(self._data_path, 'Annotations'))
        image_index = list(map(lambda x: x.replace(".xml", ""), image_index))
        if is_sort:
            image_index.sort()
        return image_index


    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        wh = tree.find('size')
        w, h = int(wh.find('width').text), int(wh.find('height').text)
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            # x1 = float(bbox.find('xmin').text) - 1
            # y1 = float(bbox.find('ymin').text) - 1
            # x2 = float(bbox.find('xmax').text) - 1
            # y2 = float(bbox.find('ymax').text) - 1

            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, w)
            y2 = min(y2, h)
            #print("bbox", [x1, y1, x2, y2])

            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            ishards[ix] = difficult

            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}


    def image_path_at(self, i):

        s = join(self._data_path, "Data", self._image_index[i] + self._image_ext)
        #print(f"Calling image_path_at {i} return {s}")
        return s

    def image_id_at(self, i):
        #print(f"calling image_id_at {i} return {self._image_index[i]}")
        return self._image_index[i]

    def default_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self, all_boxes, output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):

        aps = []
        # The PASCAL VOC metric changed in 2010
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            annopath = "/media/indoordesk/653ce34c-0c14-4427-8029-be7afe6d1989/test/Annotations/{:s}.xml"
            a = os.listdir("/media/indoordesk/653ce34c-0c14-4427-8029-be7afe6d1989/test/Annotations")
            a.sort()
            imagesetfile = "/tmp/ron.txt"
            import pickle
            with open(imagesetfile, "w") as f:
                for ff in a:
                    f.write(ff.split(".")[0])
                    f.write('\n')

            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, "/tmp/cachedir", ovthresh=0.5,
                use_07_metric=True)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')


    def _get_voc_results_file_template(self):
        return "/tmp/ron_{:s}.txt"
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        # filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        # filedir = os.path.join(self._devkit_path, 'results', 'VOC' + self._year, 'Main')
        # if not os.path.exists(filedir):
        #     os.makedirs(filedir)
        # path = os.path.join(filedir, filename)
        #return path