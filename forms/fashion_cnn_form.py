from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired, ValidationError
from flask_wtf.file import FileAllowed

class UploadForm(FlaskForm):
    file = FileField("Chọn ảnh ", validators=[
        DataRequired(message="Vui lòng chọn một ảnh."),
        FileAllowed(['jpg', 'jpeg', 'png'], 'Chỉ cho phép ảnh JPG, JPEG, PNG!')
    ])
    submit = SubmitField("Dự đoán")
