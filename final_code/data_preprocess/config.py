def obtain_train_cv(root_path, num_cv):
    cv_path = []
    for i in range(num_cv):
        cv_path_i = root_path + 'train_val_cv_%i.npz' % i
        cv_path.append(cv_path_i)
    return cv_path


TRAIN_ROOT_PATH = '../dataset/train/'
TEST_ROOT_PATH = '../dataset/test/'

TRAIN_IMG_PATH = TRAIN_ROOT_PATH + 'train_image/'
VAL_IMG_PATH = TRAIN_ROOT_PATH + 'val_image/'
TEST_IMG_PATH = TEST_ROOT_PATH + 'test_image/'

TRAIN_IMG_AUG_PATH = TRAIN_ROOT_PATH + 'train_aug_image/'
TEST_IMG_AUG_PATH = TEST_ROOT_PATH + 'test_aug_image/'

TRAIN_VAL_PATH = TRAIN_ROOT_PATH + 'train_val_data.txt'

TRAIN_LABEL_PATH = TRAIN_ROOT_PATH + 'data_train_image.txt'
VAL_LABEL_PATH = TRAIN_ROOT_PATH + 'val.txt'
TEST_INFO_PATH = TEST_ROOT_PATH + 'result_sample_1.txt'
CLASSES_NAME = TRAIN_ROOT_PATH + 'label_name.txt'


BATCH_SIZE = 128
NUM_CLASSES = 97
NUM_CV = 5

TRAIN_CV_PATH = obtain_train_cv(TRAIN_ROOT_PATH, NUM_CV)


DEVICE_ID_ALL = [0, 1, 2, 3]
