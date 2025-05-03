import os
import numpy as np
import faiss
from tqdm import tqdm
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from config import Config

def build_index():
    #dùng resnet rút trích đặc trưng
    model=ResNet50(weights="imagenet", include_top=False,pooling="avg")

    features=[] #danh sách chứa các đặc trưng
    img_paths=[] #danh sách chứa đường dẫn ảnh

    #kiểm tra xem thư mục ảnh
    if not os.path.exists(Config.PRODUCT_FOLDER):
        raise FileNotFoundError("Không có thư mục ảnh")

    #lấy danh sách các ảnh trong thư mục
    image_list=os.listdir(Config.PRODUCT_FOLDER)
    total_images=len(image_list) #đếm tổng số ảnh
    print(f"Tổng số ảnh:{total_images} ")

    #duyệt qua từng ảnh
    for idx,img_name in enumerate(tqdm(image_list)):
        img_path=os.path.join(Config.PRODUCT_FOLDER,img_name) #đừng dẫn dầy đủ ảnh
        try:
            #mở ảnh và resize kích thước ảnh
            img=image.load_img(img_path,target_size=(224,224))
            #chuyển ảnh thành mảng numpy
            img_array=image.img_to_array(img)
            #tiền xử lý dữ liệu phù hợp với yêu cầu resnet
            img_array=preprocess_input(np.expand_dims(img_array,axis=0))
            #trích xuất đặc trưng từ ảnh (dự đoán qua mô hình ResNet50)
            feat=model.predict(img_array,verbose=0).flatten()
            #thêm đặc trưng vào danh sách
            features.append(feat)
            #thêm đường dẫn ảnh vào dsnh sách
            img_paths.append(img_path)
            #in thông tin mỗi 1000 ảnh xử lý thành công
            if(idx+1)%1000==0 or idx+1==total_images:
                print(f"Đã xử lý:{idx+1}/{total_images} ảnh")

        except Exception as e:
            print(f"Lỗi xử lý ảnh: {img_path}")    


    #kiểm tra danh sách đặc trưng
    if not features:
        raise ValueError("Không có đặc nào được trích xuất, kiểm tra lại thư mục")
    
    #chuyển list features thành numpy với kiểu dữ liệu float
    features=np.array(features).astype("float32")
    #chuyển list img_paths  mảng numpy
    img_paths=np.array(img_paths)

    #đảm bảo thư mục chửa file index và ảnh đã tồn tại
    os.makedirs(os.path.dirname(Config.INDEX_PATH),exist_ok=True)

    #tạo Faiss index để lưu đặc trưng của ảnh (sử dụng khoảng cách L2)
    index=faiss.IndexFlatL2(features.shape[1])
    #thêm các đăng trưng vào index
    index.add(features)

    #lưu index vào file và lưu các dường dẫn ảnh vào file .npy
    faiss.write_index(index,Config.INDEX_PATH)
    np.save(Config.IMG_PATHS_PATH,img_paths)

    print(f"Tổng số ảnh đã xử lý thành công: {len(img_path)}/{total_images}")