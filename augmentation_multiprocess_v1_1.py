import os
import cv2
import random
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from multiprocessing import Pool, Manager, Value, Lock

image_dir = ''
origin_label = ''
output_label = ''
save_path = ''

image_range = 
static_image_index=
possibility = 

def posssibility_decorater(func):
    def random_possibility(*args, **kwargs):
        rand = random.random()
        if rand < possibility:
            func(*args, **kwargs)
    return random_possibility

def normalizer(image,i):
    labels=deepcopy(image_dict.get(str(i)+'.jpg'))
    (w,h)=image.shape[:2]
    try:
        for label in labels:            
            for i in range(0,8,2):
                label[i]=label[i]/w
                label[i+1]=label[i+1]/h
        return labels
    except:
        return []

def normalize_clip(image,i):
    image,labels,_=clip(image,i)
    (w,h)=image.shape[:2]
    try:
        for label in labels:            
            for i in range(0,8,2):
                label[i]=label[i]/w
                label[i+1]=label[i+1]/h
        return image,labels
    except:
        return image,None

def normalize_decorater(func):
    def nomalize(*args,**kwargs):
        image,labels=func(*args,**kwargs)
        (w,h)=image.shape[:2]
        for label in labels:
            for i in range(0,8,2):
                label[i]=label[i]/w
                label[i+1]=label[i+1]/h
        return image,labels
    return nomalize

def clip(image,i):
    image=deepcopy(image)
    (h,w)=image.shape[:2]
    labels=deepcopy(image_dict.get(str(i)+'.jpg',[]))
    if labels==[] or labels==[[]]:
        return image,labels,(-114,-514,h)
    scale=int(random.gauss(250,50))
    if scale<=10:
        scale=10
    elif scale>=min(w,h)*2//3:
        scale=min(w,h)*2//3
    sub_w=random.randint(0,min(w,h)-scale)
    sub_h=random.randint(0,min(w,h)-scale)
    image=image[sub_h:sub_h+scale,sub_w:sub_w+scale]
    new_labels=[]
    for label in labels:
        i=0
        process=False
        exist=False
        while i<8:
            if label[i]>=sub_w and label[i]<=sub_w+scale and label[i+1]>=sub_h and label[i+1]<=sub_h+scale and process==False:
                i=0
                process=True
                exist=True
            if process==True:
                label[i]-=sub_w
                label[i+1]-=sub_h
                if label[i]<0:
                    label[i]=0
                elif label[i]>=scale:
                    label[i]=scale-1
                if label[i+1]<0:
                    label[i+1]=0
                elif label[i+1]>=scale:
                    label[i+1]=scale-1
            i+=2
        if exist==True:
            new_labels.append(label)
    return image,new_labels,(sub_w,sub_h,scale)

def genshin_impact(label,sub_w,sub_h,scale):
    i=0
    j=0
    while i<8:
        if label[i]>=sub_w and label[i]<=sub_w+scale and label[i+1]>=sub_h and label[i+1]<=sub_h+scale :
            j=j+1
        i+=2
    if j>=3:
        return False
    else:
        return True

@normalize_decorater 
def mixup(image,mask_image,i,j):
    image=deepcopy(image)
    (h,w)=image.shape[:2]
    labels=deepcopy(image_dict.get(str(i)+'.jpg',[]))
    mask_image,mask_labels,(sub_w,sub_h,scale)=clip(mask_image,j)
    while scale>min(image.shape[:2]):
        if sub_w>0:
            mask_image,mask_labels,(sub_w,sub_h,scale)=clip(mask_image,j)
        else:
            mask_image=mask_image[0:scale//2,0:scale//2]
            mask_labels=[]
            scale=scale//2
    sub_w=random.randint(0,min(w,h)-scale)
    sub_h=random.randint(0,min(w,h)-scale)
    image[sub_h:sub_h+scale,sub_w:sub_w+scale]=mask_image
    labels=[label for label in labels if genshin_impact(label,sub_w,sub_h,scale)]
    for label in mask_labels:
        for i in range(0,8,2):
            label[i]+=sub_w
            label[i+1]+=sub_h
    if mask_labels!=[] and mask_labels!=[[]]:
        labels=labels+mask_labels
    return image,labels

image_dict={}
with open(origin_label,'r') as file:
    for line in file:
        line=list(line.strip().split())
        image_name=line[0]
        image_list=[int(x) for x in line[1:]]
        if not image_name in image_dict:
            image_dict[image_name]=[image_list]
        else:
            image_dict[image_name].append(image_list)

def rotate(image,i):
    image=deepcopy(image)
    (h,w)=image.shape[:2]
    center=(w//2,h//2)
    rotated_matrix=cv2.getRotationMatrix2D(center,random.uniform(0,360),1)
    cos=np.abs(rotated_matrix[0, 0])
    sin=np.abs(rotated_matrix[0,1])
    new_w=int((h*sin)+(w*cos))
    new_h=int((h*cos)+(w*sin))
    rotated_matrix[0,2]+=(new_w / 2)-center[0]
    rotated_matrix[1,2]+=(new_h / 2)-center[1]
    rotated_image=cv2.warpAffine(image,rotated_matrix,(new_w,new_h))
    image_name = str(i) + '.jpg'
    rotated_labels=[]
    if image_name in image_dict:
        img_labels=deepcopy(image_dict[image_name])
        for label in img_labels:
            pixel_label = [(x, y) for x, y in zip(label[::2], label[1::2])]
            rotated_label = []
            for x,y in pixel_label:
                rotated_point=(np.dot(rotated_matrix[:2, :2],np.array([x, y]))+rotated_matrix[:2,2])
                rotated_point=rotated_point/np.array([new_w,new_h])
                rotated_label.append(rotated_point.tolist())
            rotated_label=[item for sublist in rotated_label for item in sublist]
            rotated_labels.append(rotated_label)
    return rotated_image,rotated_labels

def gauss_blur(image,radius):
    image=deepcopy(image)
    radius=int(random.uniform(1,radius))
    if radius%2==0:
        radius+=1
    blurred_image=cv2.GaussianBlur(image,(radius,radius),0)
    return blurred_image
        
def gauss_noise(image,noise):
    image=deepcopy(image)
    (h,w)=image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gaussian_noise=np.random.normal(noise,15,(h,w,1)).astype('uint8')
    noisy_image=cv2.add(image,gaussian_noise)
    return noisy_image

def denoise(image,kernel_size):
    image=deepcopy(image)
    denoised_image=cv2.medianBlur(image,kernel_size)
    return denoised_image

@posssibility_decorater
def clip_api(image,i,image_index):
    clipped_image,clipped_labels=normalize_clip(image,i)
    image_save_path=os.path.join(save_path, f"{image_index}.jpg")
    cv2.imwrite(image_save_path,clipped_image)
    new_label=[str(image_index)+'.jpg']
    with open(output_label,'a') as file:
        if clipped_labels is not None:
            for label in clipped_labels:
                label.insert(0,str(image_index)+'.jpg')  
                label_str=' '.join(map(str, label))
                file.write(label_str + '\n')                

@posssibility_decorater
def mixup_api(image,i,image_index):
    j=random.randint(1,image_range)
    image_path=os.path.join(image_dir,str(j)+'.jpg')
    mask_image=cv2.imread(image_path)
    while mask_image is None:
        j=random.randint(1,image_range)
        mask_image=cv2.imread(str(j)+'.jpg')
    mixup_image,mixup_labels=mixup(image,mask_image,i,j)
    image_save_path=os.path.join(save_path, f"{image_index}.jpg")
    cv2.imwrite(image_save_path,mixup_image)
    new_label=[str(image_index)+'.jpg']
    with open(output_label,'a') as file:
        if mixup_labels is not None:
            for label in mixup_labels:
                label.insert(0,str(image_index)+'.jpg')  
                label_str=' '.join(map(str, label))
                file.write(label_str + '\n')                

@posssibility_decorater
def rotate_api(image,i,image_index):
    rotated_image,rotated_labels=rotate(image,i)
    image_save_path=os.path.join(save_path, f"{image_index}.jpg")
    cv2.imwrite(image_save_path,rotated_image)
    with open(output_label,'a') as file:
        if rotated_labels is not None:
            for label in rotated_labels:
                label.insert(0,str(image_index)+'.jpg')  
                label_str=' '.join(map(str, label))
                file.write(label_str + '\n') 

@posssibility_decorater
def gauss_blur_api(image,i,image_index):
    blurred_image=gauss_blur(image,35)
    image_save_path=os.path.join(save_path, f"{image_index}.jpg")
    cv2.imwrite(image_save_path,blurred_image)
    new_label=[str(image_index)+'.jpg']
    labels=normalizer(image,i)
    with open(output_label,'a') as file:
        if labels is not None:
            for label in labels: 
                label.insert(0,str(image_index)+'.jpg')
                label_str=' '.join(map(str, label))
                file.write(label_str + '\n')         

@posssibility_decorater
def gauss_noise_api(image,i,image_index):
    noisy_image=gauss_noise(image,100)
    image_save_path=os.path.join(save_path, f"{image_index}.jpg")
    cv2.imwrite(image_save_path,noisy_image)
    new_label=[str(image_index)+'.jpg']
    labels=normalizer(image,i)
    with open(output_label,'a') as file:
        if labels is not None:
            for label in labels:
                label.insert(0,str(image_index)+'.jpg')
                label_str=' '.join(map(str, label))
                file.write(label_str + '\n') 

@posssibility_decorater
def denoise_api(image,i,image_index):
    denoised_image=denoise(image,5)
    image_save_path=os.path.join(save_path, f"{image_index}.jpg")
    cv2.imwrite(image_save_path,denoised_image)
    new_label=[str(image_index)+'.jpg']
    labels=normalizer(image,i)
    with open(output_label,'a') as file:
        if labels is not None:
            for label in labels: 
                label.insert(0,str(image_index)+'.jpg')
                label_str=' '.join(map(str, label))
                file.write(label_str + '\n')   

def init_globals(shared_dict, empty_list, image_idx, idx_lock, progress_val):
    global image_dict
    global empty_images
    global image_index
    global lock
    global progress
    
    image_dict = shared_dict
    empty_images = empty_list
    image_index = image_idx
    lock = idx_lock
    progress = progress_val

def process_image(i):
    global image_dict
    global image_index
    global lock
    global progress

    image_path = os.path.join(image_dir, str(i) + '.jpg')
    try:
        img = cv2.imread(image_path)
        image = deepcopy(img)
        if img is None:
            raise ValueError(str(i) + '.jpg')
    except Exception as e:
        empty_images.append(e)
        return

    with lock:
        idx = image_index.value
        image_index.value += 1
    clip_api(image, i, idx)

    with lock:
        idx = image_index.value
        image_index.value += 1
    mixup_api(image, i, idx)

    with lock:
        idx = image_index.value
        image_index.value += 1
    rotate_api(image, i, idx)

    with lock:
        idx = image_index.value
        image_index.value += 1
    gauss_blur_api(image, i, idx)

    with lock:
        idx = image_index.value
        image_index.value += 1
    gauss_noise_api(image, i, idx)

    with lock:
        idx = image_index.value
        image_index.value += 1
    denoise_api(image, i, idx)

    with lock:
        progress.value += 1

def main():
    global image_dict
    global empty_images
    
    with open(origin_label, 'r') as file:
        for line in file:
            line = list(line.strip().split())
            image_name = line[0]
            image_list = [float(x) for x in line[1:]]
            if not image_name in image_dict:
                image_dict[image_name] = [image_list]
            else:
                image_dict[image_name].append(image_list)

    with Manager() as manager:
        shared_dict = manager.dict(image_dict)
        empty_list = manager.list()
        image_index = Value('i', static_image_index)
        progress = Value('i', 0)
        lock = Lock()

        init_globals(shared_dict, empty_list, image_index, lock, progress)
        
        num_processes = 4  
        pool = Pool(processes=num_processes, initializer=init_globals, initargs=(shared_dict, empty_list, image_index, lock, progress))
        
        with tqdm(total=image_range - 1, desc="Processing Images", ncols=100) as pbar:
            for _ in pool.imap_unordered(process_image, range(1, image_range)):
                try:
                    with lock:
                        pbar.update(progress.value - pbar.n)
                except:
                    pass
        
        pool.close()
        pool.join()

if __name__ == '__main__':
        main()
