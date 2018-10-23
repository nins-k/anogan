import glob
import cv2
import os
import numpy as np


def augment(img):
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    v = np.array(v, dtype=np.int16)
    v += np.random.randint(low=-15, high=15)
    v = np.array(np.clip(v, a_min=0, a_max=100), dtype=np.uint8)
    hsv = cv2.merge((h,s,v))
    hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img = cv2.flip(hsv, flipCode=1)
    
    return img
    
    
    
def adjust_length(num, length):
    diff_in_length = length - len(str(num))
    return "0"*diff_in_length + str(num)


def get_img_and_gt_names(base_dir, sub_dir):
    
    img_names = sorted(glob.glob(os.path.join(base_dir, sub_dir, "*.jpg")))
    gt_img_names = sorted(glob.glob(os.path.join(base_dir, sub_dir, "*.png")))
    
    # Check if names are in sequence
    for i in range(len(img_names)):
        assert img_names[i][:-3] == gt_img_names[i][:-3]
        
    return img_names, gt_img_names


def normalize(x):
    return (x/255)*2-1


def denormalize(x):
    return np.uint8((x+1)/2*255)


def crop_and_save_with_gt(img_names, gt_img_names, save_dir, scale_size, random_crop_range,
                          max_attempts_per_image, gt_thresh):
    
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(os.getcwd(), save_dir))

    image_set = []
    gt_image_set = []
    
    for i in range(len(gt_img_names)):
        
        image_gt = cv2.imread(gt_img_names[i])
        image = cv2.imread(img_names[i])
        
        h,w,d = image_gt.shape
        if w < h or w==h:
            w_factor = scale_size/w
            
            img_gt_r = cv2.resize(image_gt, dsize=(0,0), fx=w_factor, fy=w_factor)
            img_r = cv2.resize(image, dsize=(0,0), fx=w_factor, fy=w_factor)
            
            if(img_r.shape[0] == img_r.shape[1] == scale_size):
                
                gt_image_set.append(img_gt_r)
                image_set.append(img_r)
                
                continue
            
            h_range = img_r.shape[0] - scale_size
            gt_score = len(np.nonzero(img_gt_r)[0])
            image_thresh = (h_range // random_crop_range)+1
            images_created = 0
            
            for m in range(max_attempts_per_image):
                
                hp = np.random.randint(low=0, high=h_range)
                cropped_gt_img = img_gt_r[hp:hp+scale_size,:,:]
                cropped_gt_score = len(np.nonzero(cropped_gt_img)[0])
#                 print("Height is {}, Width is {}, Resized shape is {}, h_range is {},\
#                 hp is {}, cropped shape is {}".format(h,w,img_r.shape,h_range,hp,cropped_gt_img.shape))
                if cropped_gt_score >= gt_score*gt_thresh:
                    
                    cropped_img = img_r[hp:hp+scale_size,:,:]
                    
                    gt_image_set.append(cropped_gt_img)
                    image_set.append(cropped_img)
                    
                    images_created += 1
                    if images_created == image_thresh:
                        break

        else:
            h_factor = scale_size/h
            
            img_gt_r = cv2.resize(image_gt, dsize=(0,0), fx=h_factor, fy=h_factor)
            img_r = cv2.resize(image, dsize=(0,0), fx=h_factor, fy=h_factor)
            
            if(img_r.shape[0] == img_r.shape[1] == scale_size):
                
                gt_image_set.append(img_gt_r)
                image_set.append(img_r)
                
                continue
            
            w_range = img_r.shape[1] - scale_size
            gt_score = len(np.nonzero(img_gt_r)[0])
            image_thresh = (w_range // random_crop_range)+1
            images_created = 0
            
            for m in range(max_attempts_per_image):
                
                wp = np.random.randint(low=0, high=w_range)
                cropped_gt_img = img_gt_r[:,wp:wp+scale_size,:]
#                 print("Height is {}, Width is {}, Resized shape is {}, w_range is {},\
#                 wp is {}, cropped shape is {}".format(h,w,img_r.shape,w_range,wp,cropped_gt_img.shape))
                cropped_gt_score = len(np.nonzero(cropped_gt_img)[0])
                
                if cropped_gt_score >= gt_score*gt_thresh:
                    
                    cropped_img = img_r[:,wp:wp+scale_size,:]
                    
                    gt_image_set.append(cropped_gt_img)
                    image_set.append(cropped_img)
                    
                    images_created += 1
                    if images_created == image_thresh:
                        break
    
    for counter in range(len(image_set)):
        cv2.imwrite(os.path.join(save_dir, "{}.{}".format(adjust_length(counter, length=4), "jpg")), image_set[counter])
        cv2.imwrite(os.path.join(save_dir, "{}.{}".format(adjust_length(counter, length=4), "png")), gt_image_set[counter])
        
    return len(img_names), len(image_set)


def scale_and_random_crop(file_names, save_dir, scale_size=256, random_crop_range=50, image_count=None):
    
    '''
    Scales the image such that the smaller side is equal to scale_size
    Number of random patches extracted along the longer side = 1 + ((Longer Side Length - scale_size)//random_crop_range)
    '''
    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(os.getcwd(), save_dir))
    
    image_set = []
    
    if image_count is None:
        image_count = len(file_names)
        
    for name in file_names[:image_count]:
        image = cv2.imread(name)
        h,w,d = image.shape
        if w < h or w==h:
            w_factor = scale_size/w
            img_r = cv2.resize(image, dsize=(0,0), fx=w_factor, fy=w_factor)
            h_range = img_r.shape[0] - scale_size
            if(img_r.shape[0] == img_r.shape[1] == scale_size):
                image_set.append(img_r)
                continue

            for j in range((h_range // random_crop_range)+1):
                hp = np.random.randint(low=0, high=h_range)
                rescaled_img = img_r[hp:hp+scale_size,:,:]
                image_set.append(rescaled_img)

        else:
            h_factor = scale_size/h
            img_r = cv2.resize(image, dsize=(0,0), fx=h_factor, fy=h_factor)
            w_range = img_r.shape[1] - scale_size
            if(img_r.shape[0] == img_r.shape[1] == scale_size):
                image_set.append(img_r)
                continue

            for j in range((w_range // random_crop_range)+1):
                wp = np.random.randint(low=0, high=w_range)
                rescaled_img = img_r[:,wp:wp+scale_size,:]
                image_set.append(rescaled_img)
    
    for counter in range(len(image_set)):
        cv2.imwrite(os.path.join(save_dir, "{}.{}".format(adjust_length(counter, length=4), "jpg")), image_set[counter])
        
    return len(file_names), len(image_set)


def load_and_normalize(image_names):
    aug_random = np.random.random()
    if aug_random < 0.5:
        return normalize(np.asarray([augment(cv2.imread(name)) for name in image_names]))
    else:
        return normalize(np.asarray([cv2.imread(name) for name in image_names]))


def conv_out(input_size, kernel, stride, padding=None):
    if padding is None:
        padding = int((kernel-1)/2)
    o = int((input_size-kernel + 2*padding)/stride + 1)
    print(o, padding)
   