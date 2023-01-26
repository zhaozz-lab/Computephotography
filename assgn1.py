from skimage import io
import numpy as np
import cv2
from scipy import interpolate



def get_image(file_path):
    image = io.imread(file_path)
    image = image.astype(np.double)
    return image

def linearization(image, black, white):
    image = (image - black)*1.0/(white - black)
    image = np.clip(image,0,1.0)
    return image

def white_balance(image,r_scale, g_scale, b_scale):
    pass

def demosic(image,nums = 200):
    for i in range(0,nums):
        # print("i \n",i)
        for j in range(0,nums):
            slice_rows,slice_cols = int(image.shape[0]/nums),int(image.shape[1]/nums)
            result = np.where(image[slice_rows*i:slice_rows*(i+1),slice_cols*j:slice_cols*(j+1)] > 0.001)
            rows,cols = result[0],result[1]

            # print(len(rows))
            z = image[rows,cols]
            result = np.where(image[slice_rows*i:slice_rows*(i+1),slice_cols*j:slice_cols*(j+1)] < 0.001)
            rows_inter,cols_inter = result[0],result[1]
            # x,y = np.arange(slice_rows*i,slice_rows*(i+1),1),np.arange(slice_cols*j,slice_cols*(j+1),1)
            interpolatefunc = interpolate.interp2d(cols,rows,z,kind='linear')
            # image[rows_inter,cols_inter] = interpolatefunc(cols_inter,rows_inter)
            # print("cols",cols_inter)
            # print('rows',rows_inter)
            for i,j in zip(cols_inter,rows_inter):
                image[j,i] = interpolatefunc(i,j)
            
    return image

def test_demosic():
    image = np.array([[1,0,3,0,5],
                      [0,2,0,2,0],
                      [1,0,3,0,5]])
    # image = np.random.rand(6,8)
    # rows, cols = np.arange(0,image.shape[0],2),np.arange(0,image.shape[1],2)
    # Z = image[0::2,0::2]
    # print(np.where(image<0.001))
    # result = np.where(image>0.001)
    # rows,cols = result[0],result[1]
    # print("row and cols \n",rows,cols)
    # Z = image[rows,cols]
    print(image)
    image = demosic(image,nums = 1)
    print('\n',image)

def apply_ccm(image):
    xyz_cam = np.array([ 6988,-1384,-714,-5631,13410,2447,-1485,2204,7318])/10000.0
    rgb_xyz = np.array([0.4124564,0.3575761,0.1804375,
                        0.2126729,0.7151522,0.0721750,
                        0.0193339,0.1191920,0.9503041])
    xyz_cam = np.reshape(xyz_cam,(3,3))
    rgb_xyz = np.reshape(rgb_xyz,(3,3))
    ccm = np.dot(xyz_cam,rgb_xyz)
    ccm = np.linalg.inv(ccm)
    return np.clip(np.dot(image,ccm),0,1.0)

def gamma(image):
    result = np.where(image<=0.0031308)
    rows,cols = result[0],result[1]
    image = np.where(image>0.0031308,1.055*np.power(image,0.4098) - 0.055,image)
    
    # image[rows,cols] = image[rows,cols] * 12.92
  
    return np.clip(image,0.0,1.0)



def process_isp(file_path,bayer_pattern):
    black, white = 150, 4095
    image = get_image(file_path)
    linear_image = linearization(image,black,white)
    
    r_scale = 2.6
    g_scale = 1.000000
    b_scale = 1.4
    r,g,b = np.zeros_like(linear_image), np.zeros_like(linear_image), np.zeros_like(linear_image)
    g_r,g_b = np.zeros_like(linear_image),np.zeros_like(linear_image)
    if bayer_pattern == 'gbrg':
        r[1::2,0::2] = linear_image[1::2,0::2]
        g_r[1::2,1::2] = linear_image[1::2,1::2]
        g_b[0::2,0::2] = linear_image[0::2,0::2]
        b[0::2,1::2] = linear_image[0::2,1::2]
    elif bayer_pattern == 'grbg':    
        b[1::2,0::2] = linear_image[1::2,0::2]
        g_r[0::2,0::2] = linear_image[0::2,0::2]
        g_b[1::2,1::2] = linear_image[1::2,1::2]
        r[0::2,1::2] = linear_image[0::2,1::2]
    elif bayer_pattern == 'bggr':
        b[0::2,0::2] = linear_image[0::2,0::2]
        g_r[0::2,1::2] = linear_image[0::2,1::2]
        g_b[1::2,0::2] = linear_image[1::2,0::2]
        r[1::2,1::2] = linear_image[1::2,1::2] 
    else:
        r[0::2,0::2] = linear_image[0::2,0::2]
        g_r[0::2,1::2] = linear_image[0::2,1::2]
        g_b[1::2,0::2] = linear_image[1::2,0::2]
        b[1::2,1::2] = linear_image[1::2,1::2]
    
    g = g_r + g_b
    r = np.clip(r*r_scale,0,1.0)
    g = np.clip(g*g_scale,0,1.0)
    b = np.clip(b*b_scale,0,1.0)

    # r = demosic(r,nums = 128)
    # g = demosic(g,nums = 128)
    # b = demosic(b,nums = 128)
    r = r.reshape(r.shape[0],r.shape[1],1)
    g = g.reshape(r.shape[0],r.shape[1],1)
    b = b.reshape(r.shape[0],r.shape[1],1)
    gain = 6
    image = np.concatenate((r,g,b),axis=2)
    image = apply_ccm(image)
    # image = gamma(image)
    image = image[:,:,::-1]

    # save_path = bayer_pattern + '_nointerp.jpg'
    save_path = bayer_pattern + '.jpg'
    cv2.imwrite(save_path,image*255*gain)
if __name__ == "__main__":
    # test_demosic()
    # a = np.random.rand(100,100,1) * 0.0
    # b = np.power(a,0.1)
    # print(b)
    file_path = "./data/campus.tiff"
    # bayer_patterns = ['bggr','rggb','grbg','gbrg']
    bayer_patterns = ['rggb']
    for bayer_pattern in bayer_patterns:
        process_isp(file_path,bayer_pattern)


