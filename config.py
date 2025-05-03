import os
import random
from datetime import timedelta
from flask import url_for

class Config:
    SECRET_KEY="876ey3dtad6qwy3d6e23rrt61e176233stad"
    SQLALCHAMY_DATABASE_URI="mysql+pymysql://root@localhost/deep"
    UPLOAD_FOLDER="static/uploads"
    VIDEOS_FOLDER="static/video"
    PRODUCT_FOLDER = "static/product"
    INDEX_PATH="models/product_model.index"
    IMG_PATHS_PATH = "models/img_paths.npy"
