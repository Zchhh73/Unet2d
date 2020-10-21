import os

data_dir = "J:\\Data\\xVer2020"
img_dir = "J:\\Data\\xVer2020\\train"


def file_name_path(file_dir):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return:
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs):
            print("sub_dirs:", dirs)
            return dirs


def save_train_img2csv(file_dir, file_name):
    """
    save file path to csv,this is for segmentation
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :return:
    """
    out = open(file_name, 'w')
    count = 0
    sub_dirs = file_name_path(file_dir)
    train_dir = sub_dirs[-1]
    data_img_dir = file_dir+"/"+train_dir
    train_data_dir = file_name_path(data_img_dir)
    train_img_dir = train_data_dir[0]
    train_img_dir_path = data_img_dir + "/"+train_img_dir
    out.writelines("filename" + "\n")
    for root, dirs, files in os.walk(train_img_dir_path):
        for name in files:
            out.writelines(os.path.join(root, name) + "\n")
            count += 1
            print(count, '添加成功')
    print('添加成功，共计', count, '张img')


def save_train_mask2csv(file_dir, file_name):
    """
    save file path to csv,this is for segmentation
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :return:
    """
    out = open(file_name, 'w')
    count = 0
    sub_dirs = file_name_path(file_dir)
    train_dir = sub_dirs[-1]
    data_img_dir = file_dir + "/" + train_dir
    train_data_dir = file_name_path(data_img_dir)
    train_mask_dir = train_data_dir[-1]
    train_img_dir_path = data_img_dir + "/" + train_mask_dir
    out.writelines("filename" + "\n")
    for root, dirs, files in os.walk(train_img_dir_path):
        for name in files:
            out.writelines(os.path.join(root, name) + "\n")
            count += 1
            print(count, '添加成功')
    print('添加成功，共计', count, '张mask')


if __name__ == '__main__':
    # save_train_img2csv(data_dir, "train_img.csv")
    save_train_mask2csv(data_dir, "train_mask.csv")