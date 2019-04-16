from fastai import *
from fastai.vision import *
import torch
from fastprogress import master_bar, progress_bar
import PIL
import matplotlib.pyplot as plt
from third_party.testdataset import configdataset
from third_party.evaluate import compute_map_and_print

import pandas as pd


def open_image_cropped(fn:PathOrStr, div:bool=True, convert_mode:str='RGB', cls:type=Image)->Image:
    "Return `Image` object created from image in file `fn`."
    #fn = getattr(fn, 'path', fn)
    x = PIL.Image.open(fn).convert(convert_mode).crop(bbxs[str(fn).replace('./','')])
    x = pil2tensor(x,np.float32)
    if div: x.div_(255)
    return cls(x)

class ImageItemCrop(ImageList):
    def open(self, fn:PathOrStr)->Image:
        return open_image_cropped(fn)

def extract_vectors_batched(data,model,bs):
    model.cuda()
    model.eval()
    num_img = len(data.train_ds)
    vectors = None
    with torch.no_grad():
        idx=0
        for img_label in progress_bar(data.train_dl):
            st=idx*bs
            fin=min((idx+1)*bs, num_img)
            img,label = img_label
            out = model(img).cpu()
            if vectors is None:
                vectors = torch.zeros(num_img, out.size(1)) 
            vectors[st:fin,:] = out
            idx+=1
    return vectors

def validate_on_dataset(model, dataset_name='oxford5k', DATA_DIR='data/'):
    cfg = configdataset(dataset_name, os.path.join(DATA_DIR, 'test'))
    images = [cfg['im_fname'](cfg,i) for i in range(cfg['n'])]
    qimages = [cfg['qim_fname'](cfg,i) for i in range(cfg['nq'])]
    df = pd.DataFrame(images, columns=['Image'])
    qdf  = pd.DataFrame(qimages, columns=['qimages'])
    global bbxs
    bbxs = {qimages[i]:tuple(cfg['gnd'][i]['bbx']) for i in range(cfg['nq'])}
    BS=1
    NUM_WORKERS=8
    tfms = get_transforms(do_flip=False)
    tfms = (tfms[1],tfms[1]) #no transforms
    query_data = (ImageItemCrop.from_df(qdf,'', cols=['qimages'])
        .split_none()
        .label_const()
        .transform(tfms, resize_method=ResizeMethod.NO)
        .databunch(bs=BS, num_workers=NUM_WORKERS)
        .normalize(imagenet_stats)
       ) 
    query_data.train_dl.dl.batch_sampler.sampler = torch.utils.data.SequentialSampler(query_data.train_ds)
    query_data.train_dl.dl.batch_sampler.drop_last = False
    print ('Extracting query features...')
    query_vectors = extract_vectors_batched(query_data,model, 1)
    data = (ImageList.from_df(df,'', cols=['Image'])
            .split_none()
            .label_const()
            .transform(tfms, resize_method=ResizeMethod.NO)
            .databunch(bs=BS, num_workers=NUM_WORKERS)
            .normalize(imagenet_stats)
           ) 
    data.train_dl.dl.batch_sampler.sampler = torch.utils.data.SequentialSampler(data.train_ds)
    data.train_dl.dl.batch_sampler.drop_last = False
    print ('Extracting index features...')
    db_vectors = extract_vectors_batched(data,model,1)
    print('>> {}: Evaluating...'.format(dataset_name))
    # convert to numpy
    vecs = db_vectors.numpy()
    qvecs = query_vectors.numpy()
    # search, rank, and print
    scores = np.dot(vecs, qvecs.T)
    ranks = np.argsort(-scores, axis=0)
    compute_map_and_print(dataset_name, ranks, cfg['gnd'])
    return vecs, qvecs #If you want to check some kind of query expansion 

def get_idxs_and_dists(query_features, index_features, BS = 32):
    import faiss
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    FEAT_DIM = index_features.shape[1]
    cpu_index = faiss.IndexFlatL2(FEAT_DIM)
    cpu_index.add(index_features)
    index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
    out_dists = np.zeros((len(query_features), 100), dtype=np.float32)
    out_idxs = np.zeros((len(query_features), 100), dtype=np.int32)
    NUM_QUERY = len (query_features)
    for ind in progress_bar(range(0, len(query_features), BS)):
        fin = ind+BS
        if fin > NUM_QUERY:
            fin = NUM_QUERY
        q_descs = query_features[ind:fin]
        D, I = index.search(q_descs, 100)
        out_dists[ind:fin] = D
        out_idxs[ind:fin] = I
    return out_idxs, out_dists

def create_submission_from_features(query_features,
                                    index_features,
                                    fname,
                                    query_fnames,
                                    index_fnames):
    out_idxs, out_dists = get_idxs_and_dists(query_features, index_features, BS = 32)
    print (f'Writing {fname}')
    with open(fname, 'w') as f:
        f.write('id,images\n')
        for i in progress_bar(range(len(query_fnames))):
            ids = [index_fnames[x] for x in out_idxs[i]]
            f.write(query_fnames[i] + ',' + ' '.join(ids)+'\n')
    print('Done!')
    return
