import os
import shutil
from datetime import datetime


def traindatainit(ckpt_path, data_path, sal_stage, num=2):

    data_path = os.path.join(data_path, 'pseudo_labels')

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    if num == 1:
        for i in range(int(sal_stage)):
            label_path = os.path.join(data_path, 'label' + str(i))
            if not os.path.exists(label_path):
                os.makedirs(label_path)

    if num == 2:
        for i in range(int(sal_stage)):
            label1_path = os.path.join(data_path, 'label0_' + str(i))
            label2_path = os.path.join(data_path, 'label1_' + str(i))
            if not os.path.exists(label1_path):
                os.makedirs(label1_path)
            if not os.path.exists(label2_path):
                os.makedirs(label2_path)


def traindata_record():
    root_path = 'records/' + str(datetime.now().replace(microsecond=0)).replace(':', '-').replace(' ', '_') + '/'
    os.mkdir(root_path)

    py_train = 'trainsal.py'
    new_py_train = root_path + py_train[:-2] + 'json'
    shutil.copy(py_train, new_py_train)

    py_main = 'main.py'
    new_py_main = root_path + py_main[:-2] + 'json'
    shutil.copy(py_main, new_py_main)

    return root_path + 'train_record.json'


def valdatainit(val_path='data/val_map'):

    if not os.path.exists(val_path):
        os.makedirs(val_path)

    # if os.path.exists(val_path):
    #     for file in os.listdir(val_path):
    #         file_path = val_path + '/' + file
    #         if os.path.isfile(file_path):
    #             os.remove(file_path)


def testdatainit(sal_path, crf=False):

    if os.path.exists(sal_path):
        for file in os.listdir(sal_path):
            file_path = sal_path + '/' + file
            if os.path.isfile(file_path):
                os.remove(file_path)
        # shutil.rmtree(sal_path)

    if not os.path.exists(sal_path):
        os.makedirs(sal_path)

    if crf:
        if os.path.exists(sal_path + '_crf'):
            for file in os.listdir(sal_path + '_crf'):
                file_path = sal_path + '_crf' + '/' + file
                if os.path.isfile(file_path):
                    os.remove(file_path)
        if not os.path.exists(sal_path + '_crf'):
            os.makedirs(sal_path + '_crf')
