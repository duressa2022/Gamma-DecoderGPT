from routes.routes import chat_bp
from config import Config
from flask import Flask

app=Flask(__name__)
app.config.from_object(Config)
app.register_blueprint(chat_bp)

if __name__=="__main__":
    app.run(host=Config.HOST,port=Config.PORT,debug=Config.DEBUG)
