from flask import Blueprint
from controllers.fashion_cnn_controller import lession_cnn2


deep_router=Blueprint("deep",__name__)

#định nghĩa router
deep_router.route("/lession-ann",methods=["GET","POST"])(lession_cnn2)
