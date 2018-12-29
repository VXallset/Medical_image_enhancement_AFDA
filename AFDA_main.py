import numpy as np
import preprocessing
from matplotlib import pyplot as plt
import time
import utils

print('Start processing...')

start_time = time.time()

file_path = './demo_img/demo.raw'
img_save_path = './demo_img/'
shape_height, shape_width = 2292, 2804

raw_image = np.fromfile(file_path, dtype=np.uint16)
raw_image.shape = (shape_height, shape_width)

plt.imsave(img_save_path + 'demo.png', raw_image, cmap=plt.cm.gray)

pre_enhanced_image = 1.0 - (raw_image - np.min(raw_image)) / (np.max(raw_image) - np.min(raw_image))

roi_image = preprocessing.get_ROI(pre_enhanced_image)
ROI_time = time.time()
ROI_height, ROI_width = roi_image.shape
print('Finished extracting ROI. Time elapsed: {}s.'.format(ROI_time - start_time))

fix_orders = np.linspace(0.1, 0.9, 9)
images = np.zeros((ROI_height, ROI_width, 10))
images[..., 0] = roi_image
for i in range(1, 10):
    fix_order = fix_orders[i-1]
    images[..., i] = preprocessing.convolution_fix_order(roi_image, fix_order)
    plt.imsave(img_save_path + 'fix_order{:.1f}.png'.format(fix_order), images[..., i], cmap=plt.cm.gray)

print('Finished processing. The result images are saved in {}'.format(img_save_path))

titles =['Original ROI image'] + ['Fix-order (v = {:.1f}) enhancement'.format(v) for v in fix_orders]

utils.visualize_center_line_density((images[..., 0], images[..., 1], images[..., 2],
                                     images[..., 3], images[..., 4], images[..., 5]), titles=titles)
plt.show()


"""
#final_img = preprocessing.convolution(roi_image)

fig, axs = plt.subplots(1, 3)

#roi_image = np.uint8(255 - (roi_image - np.min(roi_image)) / (np.max(roi_image) - np.min(roi_image)) * 255)
roi_image = util.pre_processing(roi_image, 2.2)
axs[0].imshow(roi_image, cmap=plt.cm.gray)
axs[0].set_title('Original Image')


axs[1].imshow(fix_order_img, cmap=plt.cm.gray)
axs[1].set_title('Fix-Order (v={}) Enhanced Image'.format(fix_order))

adaptive_order_img = preprocessing.convolution_adaptive_order(roi_image)
print('AFDA time elapsed: {}s.'.format(time.time() - ROI_time))

#final_img = np.uint8(255 - (final_img - np.min(final_img)) / (np.max(final_img) - np.min(final_img)) * 255)
axs[2].imshow(adaptive_order_img, cmap=plt.cm.gray)
axs[2].set_title('Adaptive-Order Enhanced Image')
plt.show()
"""




