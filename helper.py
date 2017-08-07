import os,numpy as np
from os.path import dirname, exists, join, splitext
import json,scipy
from sklearn.neighbors import NearestNeighbors

class Dataset(object):
    def __init__(self, dataset_name):
        self.work_dir = dirname(os.path.realpath('__file__'))
        info_path = join(self.work_dir, 'datasets', dataset_name + '.json')
        with open(info_path, 'r') as fp:
            info = json.load(fp)
        self.palette = np.array(info['palette'], dtype=np.uint8)
        self.nns = NearestNeighbors(1,metric='euclidean')
        self.nns.fit(self.palette)  


def get_semantic_map(path,d ataset_name):
    dataset=Dataset(dataset_name)
    semantic=scipy.misc.imread(path)
    tmp=np.zeros((semantic.shape[0],semantic.shape[1],dataset.palette.shape[0]),dtype=np.float32)
    for k in range(dataset.palette.shape[0]):
        tmp[:,:,k]=np.float32((semantic[:,:,0]==dataset.palette[k,0])&(semantic[:,:,1]==dataset.palette[k,1])&(semantic[:,:,2]==dataset.palette[k,2]))
    return tmp.reshape((1,)+tmp.shape)

def get_semantic_map_nn(path,dataset_name,ignoreIndex=-1):
    dataset=Dataset(dataset_name)
    semantic=scipy.misc.imread(path)
    nns = dataset.nns.kneighbors(semantic.reshape(semantic.shape[0]*semantic.shape[1],3), 1, return_distance=False)
    hot = np.identity(dataset.palette.shape[0])
    if ignoreIndex > -1:
        hot[ignoreIndex] *= 0.0
    tmp = hot[nns]
    return tmp.reshape((1,semantic.shape[0],semantic.shape[1],dataset.palette.shape[0]))

def print_semantic_map(semantic,path,dataset_name):
    dataset=Dataset(dataset_name)
    semantic=semantic.transpose([2,3,1,0])
    prediction=np.argmax(semantic,axis=2)
    color_image=dataset.palette[prediction.ravel()].reshape((prediction.shape[0],prediction.shape[1],3))
    row,col,dump=np.where(np.sum(semantic,axis=2)==0)
    color_image[row,col,:]=0
    scipy.misc.imsave(path,color_image)
