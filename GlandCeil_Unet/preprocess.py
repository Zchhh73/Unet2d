import cv2 as cv
import os

img_path = "J:\\Data\\xVer2020\\train\\img"
img_save_path = "H:\\Data\\train\\img"

mask_path = "J:\\Data\\xVer2020\\train\\mask"
mask_save_path = "H:\\Data\\train\\mask "


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


if __name__ == "__main__":
    for root, dirs, files in os.walk(img_path):
        count = 0
        for dir in dirs:
            new_file_path = os.path.join(img_save_path, dir)
            if not os.path.exists(os.path.join(img_save_path, dir)):
                os.makedirs(new_file_path)
            single_path = os.path.join(img_path, dir)
            for file in os.listdir(single_path):
                data_path = os.path.join(single_path, file)
                img = cv.imread(data_path)
                img = cv.resize(img, (512, 512))
                cv.imwrite(os.path.join(img_save_path, dir, file), img)
                count += 1
                print("第", count, "张resize完成")
