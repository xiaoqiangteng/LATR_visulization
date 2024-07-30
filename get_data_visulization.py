import argparse
from mmcv.utils import Config, DictAction
from utils.utils import *
from experiments.ddp import *
from experiments.runner import *
import argparse
from mmcv import Config
from experiments.ddp import ddp_init
from experiments.runner import Runner
from collections import OrderedDict
from os import path as osp
import json

from dataset_generation_utils import *
from utils.MinCostFlow import SolveMinCostFlow

_file_path = "/media/data3/txq/programmings/git/LATR/work_dirs/openlane/release_iccv/latr_1000_baseline/visualization/"
_file_path_image = "/media/data3/txq/programmings/git/data/openlane/images/"
_top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])


def homographic_transformation(Matrix, x, y):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x3 homography matrix
            x (array): original x coordinates
            y (array): original y coordinates
    """
    ones = np.ones((1, len(y)))
    coordinates = np.vstack((x, y, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :]/trans[2, :]
    y_vals = trans[1, :]/trans[2, :]
    return x_vals, y_vals


def get_data_error(data_list_pred_lanes, data_list_pred_visibility_mat, data_list_gt_lanes, data_list_gt_visibility_mat,
                dist_th=1.5):
    r_lane, p_lane, c_lane = 0., 0., 0.
    x_error_close = []
    x_error_far = []
    z_error_close = []
    z_error_far = []
    
    if len(data_list_pred_lanes) <= 0 or len(data_list_gt_lanes) <= 0:
            return r_lane, p_lane, c_lane, -1, -1, -1, x_error_close, x_error_far, z_error_close, z_error_far

    gt_lanes = data_list_gt_lanes
    pred_lanes = data_list_pred_lanes
    gt_visibility_mat = data_list_gt_visibility_mat
    pred_visibility_mat = data_list_pred_visibility_mat
    cnt_gt = len(data_list_gt_lanes)
    cnt_pred = len(data_list_pred_lanes)
    

    adj_mat = np.zeros((cnt_gt, cnt_pred), dtype=int)
    cost_mat = np.zeros((cnt_gt, cnt_pred), dtype=int)
    cost_mat.fill(1000)
    num_match_mat = np.zeros((cnt_gt, cnt_pred), dtype=float)
    x_dist_mat_close = np.zeros((cnt_gt, cnt_pred), dtype=float)
    x_dist_mat_close.fill(1000.)
    x_dist_mat_far = np.zeros((cnt_gt, cnt_pred), dtype=float)
    x_dist_mat_far.fill(1000.)
    z_dist_mat_close = np.zeros((cnt_gt, cnt_pred), dtype=float)
    z_dist_mat_close.fill(1000.)
    z_dist_mat_far = np.zeros((cnt_gt, cnt_pred), dtype=float)
    z_dist_mat_far.fill(1000.)
    
    x_min = _top_view_region[0, 0]
    x_max = _top_view_region[1, 0]
    y_min = _top_view_region[2, 1]
    y_max = _top_view_region[0, 1]
    y_samples = np.linspace(y_min, y_max, num=100, endpoint=False)
    dist_th = 1.5
    ratio_th = 0.75
    close_range = 40
    close_range_idx = np.where(y_samples > close_range)[0][0]
    
    # compute curve to curve distance
    for i in range(cnt_gt):
        for j in range(cnt_pred):
            x_dist = np.abs(gt_lanes[i][:, 0] - pred_lanes[j][:, 0])
            z_dist = np.abs(gt_lanes[i][:, 2] - pred_lanes[j][:, 2])

            # apply visibility to penalize different partial matching accordingly
            both_visible_indices = np.logical_and(gt_visibility_mat[i, :] >= 0.5, pred_visibility_mat[j, :] >= 0.5)
            both_invisible_indices = np.logical_and(gt_visibility_mat[i, :] < 0.5, pred_visibility_mat[j, :] < 0.5)
            other_indices = np.logical_not(np.logical_or(both_visible_indices, both_invisible_indices))
            
            euclidean_dist = np.sqrt(x_dist ** 2 + z_dist ** 2)
            euclidean_dist[both_invisible_indices] = 0
            euclidean_dist[other_indices] = dist_th

            # if np.average(euclidean_dist) < 2*self.dist_th: # don't prune here to encourage finding perfect match
            num_match_mat[i, j] = np.sum(euclidean_dist < dist_th) - np.sum(both_invisible_indices)
            adj_mat[i, j] = 1
            # ATTENTION: use the sum as int type to meet the requirements of min cost flow optimization (int type)
            # using num_match_mat as cost does not work?
            cost_ = np.sum(euclidean_dist)
            if cost_<1 and cost_>0:
                cost_ = 1
            else:
                cost_ = (cost_).astype(int)
            cost_mat[i, j] = cost_

            # use the both visible portion to calculate distance error
            if np.sum(both_visible_indices[:close_range_idx]) > 0:
                x_dist_mat_close[i, j] = np.sum(
                    x_dist[:close_range_idx] * both_visible_indices[:close_range_idx]) / np.sum(
                    both_visible_indices[:close_range_idx])
                z_dist_mat_close[i, j] = np.sum(
                    z_dist[:close_range_idx] * both_visible_indices[:close_range_idx]) / np.sum(
                    both_visible_indices[:close_range_idx])
            else:
                x_dist_mat_close[i, j] = -1
                z_dist_mat_close[i, j] = -1

            if np.sum(both_visible_indices[close_range_idx:]) > 0:
                x_dist_mat_far[i, j] = np.sum(
                    x_dist[close_range_idx:] * both_visible_indices[close_range_idx:]) / np.sum(
                    both_visible_indices[close_range_idx:])
                z_dist_mat_far[i, j] = np.sum(
                    z_dist[close_range_idx:] * both_visible_indices[close_range_idx:]) / np.sum(
                    both_visible_indices[close_range_idx:])
            else:
                x_dist_mat_far[i, j] = -1
                z_dist_mat_far[i, j] = -1

    # solve bipartite matching vis min cost flow solver
    match_results = SolveMinCostFlow(adj_mat, cost_mat)
    match_results = np.array(match_results)

    # only a match with avg cost < self.dist_th is consider valid one
    match_gt_ids = []
    match_pred_ids = []
    match_num = 0
    if match_results.shape[0] > 0:
        for i in range(len(match_results)):
            if match_results[i, 2] < dist_th * y_samples.shape[0]:
                match_num += 1
                gt_i = match_results[i, 0]
                pred_i = match_results[i, 1]
                # consider match when the matched points is above a ratio
                if num_match_mat[gt_i, pred_i] / np.sum(gt_visibility_mat[gt_i, :]) >= ratio_th:
                    r_lane += 1
                    match_gt_ids.append(gt_i)
                if num_match_mat[gt_i, pred_i] / np.sum(pred_visibility_mat[pred_i, :]) >= ratio_th:
                    p_lane += 1
                    match_pred_ids.append(pred_i)
                x_error_close.append(x_dist_mat_close[gt_i, pred_i])
                x_error_far.append(x_dist_mat_far[gt_i, pred_i])
                z_error_close.append(z_dist_mat_close[gt_i, pred_i])
                z_error_far.append(z_dist_mat_far[gt_i, pred_i])
                
    return r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num, x_error_close, x_error_far, z_error_close, z_error_far


def plot_3D(lane_prediction, lane_visibility_prediction, lane_ground_truth, lane_visibility_ground_truth,
            data_list_P_g2im, output_file_name=None, file_name_image=None, data_list_error=None):
    if len(lane_ground_truth) == 0 or output_file_name is None:
        return
    
    image_real_test = cv2.imread(file_name_image)
    with open(file_name_image, 'rb') as f:
        image_real = (Image.open(f))
        image_real = F.crop(image_real, 0, 0, 1280 - 0, 1920)
        image_real = F.resize(image_real, size=(720, 960), interpolation=InterpolationMode.BILINEAR)
        image_real = np.array(image_real)
    
    width_image_text, height_image_text = 960, 880 - 720
    image_text = np.zeros((height_image_text, width_image_text, 3), np.uint8)
    image_text.fill(60)
    font_color, thickness, line_type = (255, 255, 255), 1, 4
    font_face, font_scale = cv2.FONT_HERSHEY_SIMPLEX, 0.5
    
    if data_list_error is not None and len(data_list_error) > 0:
        r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num, x_error_close, x_error_far, z_error_close, z_error_far = data_list_error
        x_error_close = [data for data in x_error_close if data >= 0.0]
        x_error_far = [data for data in x_error_far if data >= 0.0]
        z_error_close = [data for data in z_error_close if data >= 0.0]
        z_error_far = [data for data in z_error_far if data >= 0.0]
        x_error_close_mean, x_error_far_mean, z_error_close_mean, z_error_far_mean = -1, -1, -1, -1
        if len(x_error_close) > 0:
            x_error_close_mean = np.mean(x_error_close)
    
        if len(x_error_far) > 0:
            x_error_far_mean = np.mean(x_error_far)
    
        if len(z_error_close) > 0:
            z_error_close_mean = np.mean(z_error_close)
            
        if len(z_error_far) > 0:
            z_error_far_mean = np.mean(z_error_far)
    
        text_count = str('Count: Prediction {}, GT {}'.format(cnt_pred, cnt_gt))
        cv2.putText(image_text, text_count, (10, 30), font_face, font_scale,
                    font_color, thickness, line_type)
        text_match = str('Match number: {}'.format(match_num))
        cv2.putText(image_text, text_match, (10, 60), font_face, font_scale,
                    font_color, thickness, line_type)
        text_position_error_x = str('Position error in X: Close {}m, Far {}m'.format(round(x_error_close_mean, 4), round(x_error_far_mean, 4)))
        cv2.putText(image_text, text_position_error_x, (10, 90), font_face, font_scale,
                    font_color, thickness, line_type)
        text_position_error_z = str('Position error in Z: Close {}m, Far {}m'.format(round(z_error_close_mean, 4), round(z_error_far_mean, 4)))
        cv2.putText(image_text, text_position_error_z, (10, 120), font_face, font_scale,
                    font_color, thickness, line_type)

    # width_image, height_image = 960, 720
    width_image, height_image = 320, 880
    image = np.zeros((height_image, width_image, 3), np.uint8)
    image.fill(60)

    x_center, y_center = width_image * 0.5, height_image * 0.95
    point_size, point_color, thickness = 4, (0, 255, 0), 4
    cv2.circle(image, (int(x_center), int(y_center)), point_size, point_color, thickness)

    for i in range(12):
        cv2.line(image, (int(0.1 * width_image), int(y_center) - i * 70), (int(0.9 * width_image), int(y_center) - i * 70),
                    (150, 150, 150), 1)

        font_color, thickness, line_type = (255, 255, 255), 1, 4
        font_face, font_scale = cv2.FONT_HERSHEY_COMPLEX, 0.8
        cv2.putText(image, str(i * 10), (int(0.8 * width_image), int(y_center) - i * 70), font_face, font_scale,
                    font_color, thickness, line_type)

    for i in range(5):
        cv2.line(image, (int(x_center) - i * 70, 0), (int(x_center) - i * 70, int(y_center)), (150, 150, 150), 1)
        cv2.line(image, (int(x_center) + i * 70, 0), (int(x_center) + i * 70, int(y_center)), (150, 150, 150), 1)

    if len(lane_prediction) == 0:
        cv2.imwrite(output_file_name, image)
        return

    for i in range(len(lane_ground_truth)):
        data_list_visible_indices_ground_truth = np.logical_and(lane_visibility_ground_truth[i, :] >= 0.5, 1)
        
        if np.sum(data_list_visible_indices_ground_truth) == 0:
            return
        
        data_list_lane_ground_truth = np.array([lane_ground_truth[i][j] for j in range(len(lane_ground_truth[i])) if lane_visibility_ground_truth[i, j] > 0])
        data_lane_image_x, data_lane_image_y = [], []
        for data_lane in data_list_lane_ground_truth:
            x, y, _ = data_lane.transpose()
            x = -x

            p_0 = (int(x_center - x * 7.0), int(y_center - y * 7.0))
            # cv2.circle(image, (int(p_0[0]), int(p_0[1])), 2, (0, 0, 255), -1)
            data_lane_image_x.append(p_0[0])
            data_lane_image_y.append(p_0[1])
        data_lane_image_x = np.array(data_lane_image_x)
        data_lane_image_y = np.array(data_lane_image_y)
        
        curve_color = (255, 0, 0)
        curve_thickness = 3
        cv2.polylines(image, [np.column_stack((data_lane_image_x, data_lane_image_y))], isClosed=False, color=curve_color, thickness=curve_thickness)
        
        if data_list_P_g2im.shape[1] == 3:
            x_2d, y_2d = homographic_transformation(data_list_P_g2im, data_list_lane_ground_truth[:, 0], data_list_lane_ground_truth[:, 1])
            x_2d = x_2d.astype(np.int32)
            y_2d = y_2d.astype(np.int32)
            
            curve_color = (0, 0, 255)
            curve_thickness = 2
            cv2.polylines(image_real, [np.column_stack((x_2d, y_2d))], isClosed=False, color=curve_color, thickness=curve_thickness)
    
    for i in range(len(lane_prediction)):
        data_list_visible_indices_prediction = np.logical_and(lane_visibility_prediction[i, :] >= 0.5, 1)
        
        if np.sum(data_list_visible_indices_prediction) == 0:
            return
        
        data_list_lane_prediction = np.array([lane_prediction[i][j] for j in range(len(lane_prediction[i])) if lane_visibility_prediction[i, j] > 0])
        data_lane_image_x, data_lane_image_y = [], []
        for data_lane in data_list_lane_prediction:
            x, y, _ = data_lane.transpose()
            x = -x
            p_0 = (int(x_center - x * 7.0), int(y_center - y * 7.0))
            # cv2.circle(image, (int(p_0[0]), int(p_0[1])), 2, (0, 0, 255), -1)
            data_lane_image_x.append(p_0[0])
            data_lane_image_y.append(p_0[1])
        data_lane_image_x = np.array(data_lane_image_x)
        data_lane_image_y = np.array(data_lane_image_y)
        
        curve_color = (0, 0, 255)
        curve_thickness = 3
        cv2.polylines(image, [np.column_stack((data_lane_image_x, data_lane_image_y))], isClosed=False, color=curve_color, thickness=curve_thickness)
        
        if data_list_P_g2im.shape[1] == 3:
            x_2d, y_2d = homographic_transformation(data_list_P_g2im, data_list_lane_prediction[:, 0], data_list_lane_prediction[:, 1])
            x_2d = x_2d.astype(np.int32)
            y_2d = y_2d.astype(np.int32)
            
            curve_color = (255, 0, 0)
            curve_thickness = 2
            cv2.polylines(image_real, [np.column_stack((x_2d, y_2d))], isClosed=False, color=curve_color, thickness=curve_thickness)
    
    image_real = cv2.cvtColor(image_real, cv2.COLOR_RGB2BGR)
    merged_image = np.hstack((np.vstack((image_real, image_text)), image))
    cv2.imwrite(output_file_name, merged_image)


def get_data_error_json(data_list_error, file_name, output_file_name_error):
    if output_file_name_error is None or data_list_error is None or len(data_list_error) == 0:
        return

    r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num, x_error_close, x_error_far, z_error_close, z_error_far = data_list_error
    data_dict_json = OrderedDict()
    data_dict_json['file_name'] = file_name
    data_dict_json['r_lane'] = r_lane
    data_dict_json['p_lane'] = p_lane
    data_dict_json['c_lane'] = c_lane
    data_dict_json['cnt_gt'] = cnt_gt
    data_dict_json['cnt_pred'] = cnt_pred
    data_dict_json['match_num'] = match_num
    data_dict_json['x_error_close'] = x_error_close
    data_dict_json['x_error_far'] = x_error_far
    data_dict_json['z_error_close'] = z_error_close
    data_dict_json['z_error_far'] = z_error_far
    
    x_error_close = [data for data in x_error_close if data >= 0.0]
    x_error_far = [data for data in x_error_far if data >= 0.0]
    z_error_close = [data for data in z_error_close if data >= 0.0]
    z_error_far = [data for data in z_error_far if data >= 0.0]
    x_error_close_mean, x_error_far_mean, z_error_close_mean, z_error_far_mean = -1, -1, -1, -1
    if len(x_error_close) > 0:
        x_error_close_mean = np.mean(x_error_close)

    if len(x_error_far) > 0:
        x_error_far_mean = np.mean(x_error_far)

    if len(z_error_close) > 0:
        z_error_close_mean = np.mean(z_error_close)
        
    if len(z_error_far) > 0:
        z_error_far_mean = np.mean(z_error_far)

    data_dict_json['x_error_close_mean'] = x_error_close_mean
    data_dict_json['x_error_far_mean'] = x_error_far_mean
    data_dict_json['z_error_close_mean'] = z_error_close_mean
    data_dict_json['z_error_far_mean'] = z_error_far_mean

    write_file_json(data_dict_json, output_file_name_error)


def get_data_visulization(file_path, file_path_image):
    if file_path is None:
        return

    _, data_list_file = get_dir_and_file_list(file_path)
    
    for i in range(len(data_list_file)):
        file = data_list_file[i]
        file_name = os.path.join(file_path, file)
        print(i, file_name)
        
        data_list_file_image = file.strip().split('+')
        data_list_file_image = [data_list_file_image[0], data_list_file_image[1], data_list_file_image[2].replace('.json', '.jpg')]
        file_name_image = os.path.join(file_path_image, data_list_file_image[0], data_list_file_image[1], data_list_file_image[2])
        print(file_name_image)
        
        data_dict_lane = read_file_json(file_name)
        lane_prediction = data_dict_lane['lane_prediction']
        lane_ground_truth = data_dict_lane['lane_ground_truth']
        lane_visibility_prediction = data_dict_lane['lane_visibility_prediction']
        lane_visibility_ground_truth = data_dict_lane['lane_visibility_ground_truth']
        # transform = data_dict_lane['transform']
        transform = data_dict_lane['transform_homograpthy']
        cam_extrinsics = data_dict_lane['cam_extrinsics']
        cam_intrinsics = data_dict_lane['cam_intrinsics']
        # print(data_dict_lane.keys())
        
        data_list_pred_lanes = np.array([np.array([np.array(data_lane) for data_lane in data_list]) for data_list in lane_prediction])
        data_list_gt_lanes = np.array([np.array([np.array(data_lane) for data_lane in data_list]) for data_list in lane_ground_truth])
        data_list_pred_visibility_mat = np.array([np.array(data) for data in lane_visibility_prediction])
        data_list_gt_visibility_mat = np.array([np.array(data) for data in lane_visibility_ground_truth])

        if len(data_list_pred_lanes) == 0 or len(data_list_gt_lanes) == 0:
            continue

        additional_column = np.arange(3, 103).reshape(1, 1, -1)
        data_list_lane_prediction = np.zeros((data_list_pred_lanes.shape[0], data_list_pred_lanes.shape[1], data_list_pred_lanes.shape[2] + 1), dtype=data_list_pred_lanes.dtype)
        data_list_lane_prediction[:, :, 0] = data_list_pred_lanes[:, :, 0]
        data_list_lane_prediction[:, :, 2] = data_list_pred_lanes[:, :, 1]
        data_list_lane_prediction[:, :, 1] = additional_column

        data_list_lane_ground_truth = np.zeros((data_list_gt_lanes.shape[0], data_list_gt_lanes.shape[1], data_list_gt_lanes.shape[2] + 1), dtype=data_list_gt_lanes.dtype)
        data_list_lane_ground_truth[:, :, 0] = data_list_gt_lanes[:, :, 0]
        data_list_lane_ground_truth[:, :, 2] = data_list_gt_lanes[:, :, 1]
        data_list_lane_ground_truth[:, :, 1] = additional_column

        data_list_P_g2im = np.array([np.array(data_lane) for data_lane in transform])
        data_list_cam_extrinsics = np.array([np.array(data_lane) for data_lane in cam_extrinsics])
        data_list_cam_intrinsics = np.array([np.array(data_lane) for data_lane in cam_intrinsics])
        
        r_lane, p_lane, c_lane, cnt_gt, cnt_pred, \
            match_num, x_error_close, x_error_far, \
                z_error_close, z_error_far = get_data_error(data_list_lane_prediction,
                                                            data_list_pred_visibility_mat,
                                                            data_list_lane_ground_truth,
                                                            data_list_gt_visibility_mat)
        print(r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num, x_error_close, x_error_far, z_error_close, z_error_far)
        data_list_error = [r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num, x_error_close, x_error_far, z_error_close, z_error_far]
        
        output_file_path_error = osp.join(_file_path, 'results_error/')
        if not os.path.exists(output_file_path_error):
            os.makedirs(output_file_path_error)
            
        output_file_name_error = os.path.join(output_file_path_error, file)
        get_data_error_json(data_list_error, file, output_file_name_error)
        
        output_file_path = osp.join(_file_path, 'results/')
        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)
        
        if data_list_error is not None and len(data_list_error) > 0:
            r_lane, p_lane, c_lane, cnt_gt, cnt_pred, match_num, x_error_close, x_error_far, z_error_close, z_error_far = data_list_error
            x_error_close = [data for data in x_error_close if data >= 0.0]
            x_error_far = [data for data in x_error_far if data >= 0.0]
            z_error_close = [data for data in z_error_close if data >= 0.0]
            z_error_far = [data for data in z_error_far if data >= 0.0]
            x_error_close_mean, x_error_far_mean, z_error_close_mean, z_error_far_mean = -1, -1, -1, -1
            if len(x_error_close) > 0:
                x_error_close_mean = np.mean(x_error_close)
        
            if len(x_error_far) > 0:
                x_error_far_mean = np.mean(x_error_far)
        
            if len(z_error_close) > 0:
                z_error_close_mean = np.mean(z_error_close)
                
            if len(z_error_far) > 0:
                z_error_far_mean = np.mean(z_error_far)
                
            if x_error_close_mean > 0 and x_error_far_mean > 0 and x_error_close_mean <= x_error_far_mean:
                continue
        else:
            continue
            
        output_file_name = os.path.join(output_file_path, file.replace('.json', '.jpg'))
        plot_3D(data_list_lane_prediction,
                data_list_pred_visibility_mat,
                data_list_lane_ground_truth,
                data_list_gt_visibility_mat,
                data_list_P_g2im,
                output_file_name=output_file_name,
                file_name_image=file_name_image,
                data_list_error=data_list_error)


if __name__ == "__main__":
    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})
    get_data_visulization(_file_path, _file_path_image)
    