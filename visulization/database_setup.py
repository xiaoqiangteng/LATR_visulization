
from app import db, Image, app
from os import path as osp
import os

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


def get_data_image(file_path):
    file_path = osp.join(file_path, 'results/')
    _, data_list_file = get_dir_and_file_list(file_path)
    
    data_list_image = []
    for file in data_list_file:
        file_name = osp.join(file_path, file)
        data_list_image.append(Image(image_path='static/images/results/{}'.format(file), position_deviation='小', scene='市区'))
        
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
