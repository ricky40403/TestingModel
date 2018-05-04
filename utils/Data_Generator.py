"""Pascal VOC Segmenttion Generator."""
from __future__ import unicode_literals
import os
import numpy as np
from keras.utils.np_utils import to_categorical
import random
import scipy.misc
import scipy.ndimage
import tensorflow as tf
from scipy import ndimage
from keras import backend as K
from keras.preprocessing.image import (
    ImageDataGenerator,
    Iterator,
    load_img,
    img_to_array,
    pil_image,
    array_to_img)


############################################################
#  Boxes
############################################################

def generate_anchors(scales, ratios, i):
    
    BACKBONE_SIZE = [56,28,14,7]
    BACKBONE_STRIDES = [4, 8, 16, 32]
    
    # Get all combinations of scales and ratios
    scales, ratios = np.meshgrid(np.array(scales), np.array(ratios))
    scales = scales.flatten()
    ratios = ratios.flatten()

    # Enumerate heights and widths from scales and ratios
    heights = scales / np.sqrt(ratios)
    widths = scales * np.sqrt(ratios)

    # Enumerate shifts in feature space
    shifts_y = np.arange(0, BACKBONE_SIZE[i]) * BACKBONE_STRIDES[i]
    shifts_x = np.arange(0, BACKBONE_SIZE[i]) * BACKBONE_STRIDES[i]
    shifts_x, shifts_y = np.meshgrid(shifts_x, shifts_y)
    
    # Enumerate combinations of shifts, widths, and heights
    box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
    box_heights, box_centers_y = np.meshgrid(heights, shifts_y)

    # Reshape to get a list of (y, x) and a list of (h, w)
    box_centers = np.stack(
        [box_centers_y, box_centers_x], axis=2).reshape([-1, 2])
    box_sizes = np.stack([box_heights, box_widths], axis=2).reshape([-1, 2])

    # Convert to corner coordinates (y1, x1, y2, x2)
    boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                            box_centers + 0.5 * box_sizes], axis=1)
    
    return boxes

def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.
    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]
    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    rpn_match_class = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    #     rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))
    rpn_bbox = np.zeros((12495, 2))
    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    print(anchor_iou_max.shape)
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    rpn_match_class[gt_iou_argmax] = gt_class_ids[np.arange(gt_iou_argmax.shape[0])]

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
#         rpn_bbox[ix] = [
#             (gt_center_y - a_center_y) / a_h,
#             (gt_center_x - a_center_x) / a_w,
#             np.log(gt_h / a_h),
#             np.log(gt_w / a_w),
#         ]
        # only predict  y and x 
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w
        ]
        # Normalize
#         rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        rpn_bbox[ix] /= [0.1, 0.1]
        ix += 1

    return rpn_match, rpn_bbox


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

    

############################################################
#  Image Handling
############################################################

def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = scipy.misc.imresize(
            image, (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding

def mold_image(images, config):
    """Takes RGB images with 0-255 values and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL

############################################################
#  Maask Handling
############################################################

def resize_mask(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    h, w = mask.shape[:2]
    mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.
    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta



def load_image_gt(dataset, config, image_id, augment=False,use_mini_mask=False,use_background = False):

    """Load and return ground truth data for an image .
    dataset : Using uitls.dataset
    augment: If true, apply random image augmentation. Currently, only
        horizontal flipping is offered.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    class_ids: [instance_count] Integer class IDs
    mask: [height, width, class_num]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """

    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    shape = image.shape
    image, window, scale, padding = resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)
    mask = resize_mask(mask, scale, padding)
    
    class_num = dataset.num_classes
#     if use_background :
#         class_num = class_num + 1

    # Random horizontal flips.
    if augment:
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    #if mini_mask resize mask size
    # init ground_true_mask

    if use_mini_mask:
        y_mask = np.zeros(config.MINI_MASK_SHAPE+(class_num,))
        y_all_mask = np.zeros(config.MINI_MASK_SHAPE)
        y_edge = np.zeros(config.MINI_MASK_SHAPE+(1,))
    else:
        y_mask = np.zeros((shape[0],shape[1],class_num))
        y_all_mask = np.zeros((shape[0],shape[1]))
        y_edge = np.zeros(config.MINI_MASK_SHAPE+(1,))
        
    
    # generate bbox from mask
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    bbox = extract_bboxes(mask)
        
    
    if use_mini_mask:
        for i in range(0,class_ids.shape[0]):
            tmp = mask[:,:,i]
            if use_mini_mask:
                tmp = scipy.misc.imresize(tmp, config.MINI_MASK_SHAPE)

            y_mask[:,:,class_ids[i]] = np.logical_or(y_mask[:,:,class_ids[i]],tmp)
            y_all_mask = y_all_mask + tmp
        y_all_mask = np.where(y_all_mask>0, 1, 0)
        
        
#     for i in range(0,mask.shape[-1]):
#         tmp_mask = mask[:,:,i]
#         struct = ndimage.generate_binary_structure(2, 2)
#         erode = ndimage.binary_erosion(tmp_mask, struct)
#         edges = (tmp_mask!=erode).astype(float)
#         y_edge[:,:,0] += edges
        
#     y_edge = np.where(np.greater(y_edge, 0), 1,0)
    
    
            
    #generate background
    #in CoCo Background is class 0
    if use_background :
        if use_mini_mask:
            background_mask = np.zeros(config.MINI_MASK_SHAPE)
        else:
            background_mask = np.zeros((shape[0],shape[1]))
            
        #resize mask
        if use_mini_mask:
            background_mask = scipy.misc.imresize(background_mask, config.MINI_MASK_SHAPE)

        for i in range(0,mask.shape[2]):
            tmp = mask[:,:,i]
            # resize mask
            if use_mini_mask:
                tmp = scipy.misc.imresize(tmp, config.MINI_MASK_SHAPE)
            background_mask = np.logical_or(background_mask,tmp)

        # xor 
        background_mask = np.logical_xor(background_mask,1)

        y_mask[:,:,0] = background_mask
    
    
    
    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1
    
    
    original_shape = image.shape
    # Image meta data
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids)
    
    return image, image_meta, class_ids, bbox, y_mask.astype(float), y_all_mask.astype(float)

def data_generator(dataset_name, dataset, config, shuffle=True, augment=True,batch_size=1):
    
    b = 0  # batch item index
    image_index = -1
    
    # set dataset class number
    # image_ids is filename in VOC2012
    # image_ids is image_id in COCO
    if dataset_name == "VOC":
        class_num = 21
        image_ids = np.copy(dataset.filenames)
    else:
        class_num = 81
        image_ids = np.copy(dataset.image_ids)
    
    anchors = generate_anchor_boxes()
    
    
    error_count = 0
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)
            
            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]

            if dataset_name == "VOC":
                image = dataset.load_img(image_id)
                mask = dataset.load_seg(image_id)
    
                image, window, scale, padding = resize_image(
                    image,
                    min_dim=config.IMAGE_MIN_DIM,
                    max_dim=config.IMAGE_MAX_DIM,
                    padding=config.IMAGE_PADDING)
                mask = resize_mask(mask, scale, padding)
                
                mask = to_categorical(mask, class_num).reshape(224,224,class_num)
                
            else:
                image, gt_class_ids, gt_boxes, masks, gt_all_masks,gt_edges = \
                load_image_gt(dataset, config, image_id, augment=augment,
                          use_mini_mask=config.USE_MINI_MASK)
                
                # Skip images that have no instances. This can happen in cases
                # where we train on a subset of classes and the image doesn't
                # have any of the classes we care about.
                if not np.any(gt_class_ids > 0):
                    continue
                    
            # RPN Targets
            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                                    gt_class_ids, gt_boxes, config)
            
            
                
            # Init batch arrays
            if b == 0:         
                batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)
                batch_edge = np.zeros((batch_size, 224, 224, 81))
                batch_masks = np.zeros((batch_size, 224, 224, class_num))
                batch_back = np.zeros((batch_size,224, 224, 2))
                batch_rpn_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, 224*224, 4], dtype=rpn_bbox.dtype)
                    
               
            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_masks[b, :, :, :] = mask
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            
            
            
            # Semantic Edge
            batch_edge[b] = batch_edge
            
            
            
            # back_edge
#             batch_backedge = np.zeros((224,224,81))
#             struct = ndimage.generate_binary_structure(2, 2)
#             tmp_background = np.zeros((config.MINI_MASK_SHAPE[0],config.MINI_MASK_SHAPE[1]))
#             for i in range(0,mask.shape[-1]):         
#                 tmp_edge = mask[:,:,i]
#                 erode = ndimage.binary_erosion(tmp_edge, struct)
#                 edges = (tmp_edge!=erode).astype(float)
#                 batch_backedge[:,:,i] = edges
#                 tmp_background += edges
                
#             batch_backedge[:,:,0] = np.where(tmp_background>0,1,0)
#             # enhance edge by back_edge
#             if dataset_name == "COCO":
#                 batch_edge[b,:,:,0] = batch_backedge[:,:,0] + gt_edges[:,:,0]
#                 batch_edge[b,:,:,:] = np.where(batch_edge[b,:,:,:]>0, 1 ,0)
            
#             # make edge
#             struct = ndimage.generate_binary_structure(2, 2)
#             tmp_background = np.zeros((224,224))
#             for i in range(0,mask.shape[-1]):         
#                 tmp_edge = mask[:,:,i]
#                 erode = ndimage.binary_erosion(tmp_edge, struct)
#                 edges = (tmp_edge!=erode).astype(float)
#                 tmp_background += edges

            # add edge info
#             batch_edge[b,:,:,0] = np.where(tmp_background>0,1,0)
#                 batch_edge[0,:,:,1] = 1-batch_edge[0,:,:,0]
            # add background info
            batch_back[b,:,:,0] = 1 - mask[:,:,0]
            batch_back[b,:,:,1] = mask[:,:,0]


            B = batch_back
            E = batch_edge
            P = batch_masks
            
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images,batch_rpn_match, batch_rpn_bbox]
                outputs = []


                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise




# PASCAL Image Loader
class PASCAL_ImageSetLoader(object):
    """Helper class to load image data into numpy arrays."""

    def __init__(self, image_set, image_dir, label_dir, target_size=(224, 224),
                 image_format='jpg', color_mode='rgb', label_format='png',
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='jpg'):
        """Init."""
        if data_format is None:
            data_format = K.image_data_format()
        self.data_format = data_format
        self.target_size = tuple(target_size)

        if not os.path.exists(image_set):
            raise IOError('Image set {} does not exist. Please provide a'
                          'valid file.'.format(image_set))
        self.filenames = np.loadtxt(image_set, dtype=bytes)
        try:
            self.filenames = [fn.decode('utf-8') for fn in self.filenames]
        except AttributeError as e:
            print(str(e), self.filenames[:5])
        if not os.path.exists(image_dir):
            raise IOError('Directory {} does not exist. Please provide a '
                          'valid directory.'.format(image_dir))
        self.image_dir = image_dir
        if label_dir and not os.path.exists(label_dir):
            raise IOError('Directory {} does not exist. Please provide a '
                          'valid directory.'.format(label_dir))
        self.label_dir = label_dir

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}
        self.image_format = image_format
        if self.image_format not in white_list_formats:
            raise ValueError('Invalid image format:', image_format,
                             '; expected "png", "jpg", "jpeg" or "bmp"')
        self.label_format = label_format
        if self.label_format not in white_list_formats:
            raise ValueError('Invalid image format:', label_format,
                             '; expected "png", "jpg", "jpeg" or "bmp"')

        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        self.grayscale = self.color_mode == 'grayscale'

        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

    def load_img(self, fn):
        """Image load method.

        # Arguments
            fn: filename of the image (without extension suffix)
        # Returns
            arr: numpy array of shape self.image_shape
        """
        img_path = os.path.join(self.image_dir,
                                '{}.{}'.format(fn,
                                               self.image_format))
        if not os.path.exists(img_path):
            raise IOError('Image {} does not exist.'.format(img_path))
        img = load_img(img_path)
        x = img_to_array(img, data_format=self.data_format)

        return x
    
    def load_seg(self, fn):
        """Segmentation load method.

        # Arguments
            fn: filename of the image (without extension suffix)
        # Returns
            arr: numpy array of shape self.target_size
        """
        label_path = os.path.join(self.label_dir,
                                  '{}.{}'.format(fn, self.label_format))
        img = pil_image.open(label_path)
        if self.target_size:
            wh_tuple = (self.target_size[1], self.target_size[0])
#         if img.size != wh_tuple:
#             img = img.resize(wh_tuple)
        y = img_to_array(img, self.data_format)
        y[y == 255] = 0
        
        return y

    def save(self, x, y, index):
        """Image save method."""
        img = array_to_img(x, self.data_format, scale=True)
        mask = array_to_img(y, self.data_format, scale=True)
        img.paste(mask, (0, 0), mask)

        fname = 'img_{prefix}_{index}_{hash}.{format}'.format(
            prefix=self.save_prefix,
            index=index,
            hash=np.random.randint(1e4),
            format=self.save_format)
        img.save(os.path.join(self.save_to_dir, fname))
