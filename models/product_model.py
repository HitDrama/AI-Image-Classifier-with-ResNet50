import os
import numpy as np
import faiss
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from config import Config

class ImageSearch:
    def __init__(self):
        # Khởi tạo model ResNet50 (không dùng tầng phân loại, pooling='avg' để lấy vector đặc trưng)
        self.model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

    def dactrung(self, img_path):
        """Trích xuất đặc trưng ảnh"""
        img = image.load_img(img_path, target_size=(224, 224))  # Resize ảnh về đúng kích thước
        img_array = image.img_to_array(img)                     # Chuyển ảnh thành mảng numpy
        img_array = preprocess_input(np.expand_dims(img_array, axis=0))  # Tiền xử lý như mô hình yêu cầu
        return self.model.predict(img_array, verbose=0).flatten()        # Dự đoán & flatten thành vector

    def build_index(self):
        features, img_paths = [], []

        for img_name in os.listdir(Config.PRODUCT_FOLDER):
            img_path = os.path.join(Config.PRODUCT_FOLDER, img_name)
            try:
                feat = self.dactrung(img_path)  # Trích xuất đặc trưng
                features.append(feat)           # Lưu vector đặc trưng
                img_paths.append(img_path)      # Lưu đường dẫn ảnh
            except:
                continue  # Bỏ qua ảnh lỗi

        features = np.array(features)  # Chuyển thành mảng NumPy

        if len(features.shape) == 1:  # Nếu features có 1 chiều, thêm chiều thứ hai
            features = np.expand_dims(features, axis=0)

        print("Shape of features:", features.shape)  # In kích thước của features

        # Chuyển features sang float32 để FAISS xử lý
        features = features.astype('float32')
        img_paths = np.array(img_paths)

        index = faiss.IndexFlatL2(features.shape[1])  # Tạo FAISS index theo L2 distance
        index.add(features)  # Thêm các đặc trưng vào index

        faiss.write_index(index, Config.INDEX_PATH)  # Ghi index vào file
        np.save(Config.IMG_PATHS_PATH, img_paths)  # Ghi đường dẫn ảnh


    def themanh(self, img_path):
        """Thêm 1 ảnh mới vào index"""
        if not os.path.exists(Config.INDEX_PATH) or not os.path.exists(Config.IMG_PATHS_PATH):
            self.build_index()  # Nếu chưa có index thì tạo mới

        index = faiss.read_index(Config.INDEX_PATH)            # Load FAISS index hiện có
        img_paths = np.load(Config.IMG_PATHS_PATH).tolist()    # Load mảng đường dẫn ảnh (chuyển sang list)

        try:
            feat = self.extract_feature(img_path)              # Trích xuất đặc trưng ảnh mới
        except:
            return

        index.add(np.expand_dims(feat, axis=0))                # Thêm vector mới vào index
        img_paths.append(img_path)                             # Thêm đường dẫn mới

        faiss.write_index(index, Config.INDEX_PATH)            # Ghi lại index
        np.save(Config.IMG_PATHS_PATH, np.array(img_paths))    # Ghi lại đường dẫn ảnh

    def timanh(self, query_feature, top_k=5):
        """Tìm top K ảnh gần nhất với ảnh query"""
        index = faiss.read_index(Config.INDEX_PATH)            # Load FAISS index
        img_paths = np.load(Config.IMG_PATHS_PATH)             # Load danh sách đường dẫn ảnh

        D, I = index.search(np.array([query_feature]).astype('float32'), top_k)  # Tìm top-k gần nhất

        result_paths = []
        for path in img_paths[I[0]]:                           # Lấy đường dẫn tương ứng với kết quả
            filename = os.path.basename(path)                  # Lấy tên file
            static_path = f'static/product/{filename}'         # Chuyển về đường dẫn dùng cho web
            result_paths.append(static_path)

        return result_paths
