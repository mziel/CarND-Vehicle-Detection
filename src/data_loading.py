import numpy as np
import matplotlib.image as mpimg
import glob
from sklearn.model_selection import train_test_split


# Read in cars and notcars
cars = glob.glob('training_large/vehicles/*/*.png')
notcars = glob.glob('training_large/non-vehicles/*/*.png')


def read_images(car_paths, notcar_paths):
    cars_img = [mpimg.imread(img)for img in car_paths]
    notcars_img = [mpimg.imread(img) for img in notcar_paths]
    return cars_img, notcars_img


def xy_train_test(x_list, y_list, rand_state=np.random.randint(0, 100)):
    y = np.hstack((np.ones(len(x_list)), np.zeros(len(y_list))))
    indices = np.arange(y.shape[0])
    indices_train, indices_test, y_train, y_test = train_test_split(indices, y, test_size=0.2, random_state=rand_state)
    return indices_train, indices_test, y_train, y_test, rand_state
