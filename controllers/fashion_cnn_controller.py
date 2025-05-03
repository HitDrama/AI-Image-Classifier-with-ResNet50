from flask import render_template, request, flash
from werkzeug.utils import secure_filename
from models.product_model import ImageSearch
from forms.fashion_cnn_form import UploadForm
import os
from config import Config

def lession_cnn2():
    form = UploadForm()
    model=ImageSearch()
    if form.validate_on_submit():
        file = request.files.get("file")
        filename = secure_filename(file.filename)
        filepath = os.path.join(Config.PRODUCT_FOLDER, filename)
        file.save(filepath)

        # Nếu bạn muốn thêm ảnh đó vào FAISS luôn:
        model.themanh(filepath)
        feat = model.dactrung(filepath)
        results = model.timanh(feat)

        return render_template('lession_cnn2_ketqua.html', query_img=filepath, similar_images=results)

    return render_template('lession_cnn2.html', form=form)
