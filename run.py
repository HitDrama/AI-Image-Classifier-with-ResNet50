from flask import Flask
from config import Config
from routes import deep_router

app= Flask(__name__)
app.config.from_object(Config)

#dăng ký blueprint
app.register_blueprint(deep_router)

if __name__=="__main__":
    app.run(debug=True)