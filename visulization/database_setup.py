
from app import db, Image, app
from os import path as osp
import os
import json

_file_path = "/media/data3/txq/programmings/git/LATR/work_dirs/openlane/release_iccv/latr_1000_baseline/visualization/"


def get_dir_and_file_list(file_path):
    dir_list, file_name_list = [], []
    for _, dirs, files in os.walk(file_path):
        for dir in dirs:
            dir_list.append(dir)

        break

    for root, dirs, files in os.walk(file_path):
        for file_name in files:
            file_name_list.append(file_name)

        break

    return dir_list, file_name_list


def read_file_json(file_name):
    with open(file_name, 'r') as f:
        data_dict = json.load(f)

    return data_dict


def get_data_image(file_path):
    file_path_image = osp.join(file_path, 'results/')
    file_path_error = osp.join(file_path, 'results_error/')
    _, data_list_file = get_dir_and_file_list(file_path_image)
    
    data_list_image = []
    for file in data_list_file:
        file_name = osp.join(file_path_error, file)
        file_name_error = file_name.replace('.jpg', '.json')
        
        data_dict_error = read_file_json(file_name_error)

        r_lane = data_dict_error['r_lane']
        p_lane = data_dict_error['p_lane']
        c_lane = data_dict_error['c_lane']
        cnt_gt = data_dict_error['cnt_gt']
        cnt_pred = data_dict_error['cnt_pred']
        match_num = data_dict_error['match_num']
        x_error_close = data_dict_error['x_error_close']
        x_error_far = data_dict_error['x_error_far']
        z_error_close = data_dict_error['z_error_close']
        z_error_far = data_dict_error['z_error_far']
        x_error_close_mean = data_dict_error['x_error_close_mean']
        x_error_far_mean = data_dict_error['x_error_far_mean']
        z_error_close_mean = data_dict_error['z_error_close_mean']
        z_error_far_mean = data_dict_error['z_error_far_mean']
        
        x_error_close_mean_large, x_error_far_mean_large = 0, 0
        z_error_close_mean_large, z_error_far_mean_large = 0, 0
        x_error_close_mean_20, x_error_far_mean_20 = 0, 0
        z_error_close_mean_20, z_error_far_mean_20 = 0, 0
        
        if x_error_close_mean > 0 and x_error_far_mean > 0:
            if x_error_close_mean > x_error_far_mean:
                x_error_close_mean_large = 1
            else:
                x_error_far_mean_large = 1
                
            if x_error_close_mean > 0.2:
                x_error_close_mean_20 = 1
                
            if x_error_far_mean > 0.2:
                x_error_far_mean_20 = 1
        
        if x_error_close_mean > 0.2:
            x_error_close_mean_20 = 1
            
        if x_error_far_mean > 0.2:
            x_error_far_mean_20 = 1
    
        if z_error_close_mean > 0 and z_error_far_mean > 0:
            if z_error_close_mean > z_error_far_mean:
                z_error_close_mean_large = 1
            else:
                z_error_far_mean_large = 1
            
        if z_error_close_mean > 0.2:
            z_error_close_mean_20 = 1
            
        if z_error_far_mean > 0.2:
            z_error_far_mean_20 = 1
    
        data_image = Image(image_path='static/images/results/{}'.format(file),
                           x_error_close_mean=x_error_close_mean,
                           x_error_far_mean=x_error_far_mean,
                           z_error_close_mean=z_error_close_mean,
                           z_error_far_mean=z_error_far_mean,
                           x_error_close_mean_large=x_error_close_mean_large,
                           x_error_far_mean_large=x_error_far_mean_large,
                           z_error_close_mean_large=z_error_close_mean_large,
                           z_error_far_mean_large=z_error_far_mean_large,
                           x_error_close_mean_20=x_error_close_mean_20,
                           x_error_far_mean_20=x_error_far_mean_20,
                           z_error_close_mean_20=z_error_close_mean_20,
                           z_error_far_mean_20=z_error_far_mean_20,
                           scene='市区')
        
        data_list_image.append(data_image)
        
    return data_list_image


def setup_database(data_list_image):
    # Create the database and the table
    with app.app_context():
        db.create_all()
        
        db.session.bulk_save_objects(data_list_image)
        db.session.commit()

if __name__ == "__main__":
    data_list_image = get_data_image(_file_path)
    
    setup_database(data_list_image)
