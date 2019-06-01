import os

from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
# from flask_bootstrap import Bootstrap

# app initialization
application = Flask(__name__)
# bootstrap = Bootstrap(application)
application.secret_key = os.urandom(24)  # for CSRF
application.secret_key = bytearray([1] * 24)

# config and initialize a db
application.config.from_object(Config)
db = SQLAlchemy(application)
db.create_all()
db.session.commit()

# config login manager
login_manager = LoginManager()
login_manager.init_app(application)

from app import routes
from app import classes

# kill -9 `ps aux |grep gunicorn |grep app | awk '{ print $2 }'`
