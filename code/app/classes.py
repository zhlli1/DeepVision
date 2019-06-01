from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash

from app import db, login_manager

from flask_wtf import FlaskForm, RecaptchaField
from wtforms import StringField, TextField
from wtforms.validators import DataRequired, Email, Length
from sqlalchemy.schema import ForeignKey


class User(db.Model, UserMixin):
    '''
    This class inherits from db.Model, UserMixin and is used to
    create user instances with multiple associated properties
    such as username and email. It is also used by ``SQLALCHEMY``
    to store user records in the data base.
    '''
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(80), unique=True, nullable=False)
    companyname = db.Column(db.String(80), nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)


    def __init__(self, username, email, companyname, password):
        '''
        Set the main attributes of a user.

        :param username: (str) username
        :param email: (str) email
        :param companyname: (str) commpany name
        :param password: (str) password  (not hashed)
        '''
        self.username = username
        self.email = email
        self.companyname = companyname
        self.set_password(password)

    def set_password(self, password):
        '''
        Transforms the introduced password to it's hashed version and
        stores it in the ```password_hash``` attribute.
        :param password: (str) transform the introduced password \ 
        to it's hashed version to store it.
        :return: None
        '''
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        '''
        Check if the introduced password matches the one in the records.

        :param password: (str) password (not hashed)
        :return: (bool) True/False if it do/do not match.
        '''
        return check_password_hash(self.password_hash, password)


class Project(db.Model):
    '''
    This class inherits from db.Model and is used to create project
    instances with multiple associated properties such as project owner
    and creation date. It is also used by ``SQLALCHEMY`` to store project
    records in the data base.
    '''
    project_id = db.Column(
        db.Integer, primary_key=True, unique=True, autoincrement=True)
    project_name = db.Column(db.String(100), nullable=False)
    project_owner_id = db.Column(db.Integer, nullable=False)
    project_creation_date = db.Column(db.DateTime, nullable=False)
    last_train_asp_ratio = db.Column(db.Float, nullable=True)

    def __init__(self, project_name, project_owner_id, last_train_asp_ratio=None):
        '''
        Set the main attributes of a project.

        :param project_name: (str) project name        
        :param project_owner_id: (str) user ``id`` of the project's owner
        :param last_train_asp_ratio: (float) ratio of width to heigh used in last training
        '''
        self.project_name = project_name
        self.project_owner_id = project_owner_id
        self.project_creation_date = datetime.utcnow()
        self.last_train_asp_ratio = last_train_asp_ratio


class Label(db.Model):
    '''
    This class inherits from db.Model and is used to create labels
    instances with multiple associated properties such as label id and
    project id. It is also used by ``SQLALCHEMY`` to store label records
    in the data base.

    It's main functionality is mapping label names to labels index so they
    can change the names if they want to without affecting system.
    '''
    label_id = db.Column(
        db.Integer, primary_key=True, unique=True, autoincrement=True)
    project_id = db.Column(
        db.Integer, ForeignKey(Project.project_id), nullable=False)
    label_name = db.Column(db.String(80), nullable=False)
    label_index = db.Column(db.Integer, nullable=True)

    def __init__(self, project_id, label_name, label_index=None):
        '''
        Set the main attributes of a label.

        :param project_id: (str) Id of the project where the label belongs.
        :param label_name: (str) Name of the label
        :param label_index: (int) label used when training
        '''
        self.project_id = project_id
        self.label_name = label_name
        self.label_index = label_index


class User_Project(db.Model):
    '''
    This class inherits from db.Model and is used to relate projects to users.
    To what projects do a user have access to ? .

    It is also used by ``SQLALCHEMY`` to store those relations records in the
    data base.
    '''
    user_project_id = db.Column(
        db.Integer, primary_key=True, unique=True, autoincrement=True)
    user_id = db.Column(
        db.Integer, ForeignKey(User.id), nullable=False)
    project_id = db.Column(
        db.Integer, ForeignKey(Project.project_id), nullable=False)

    def __init__(self, user_id, project_id):
        '''
        Set the attributes of for the relations between
        user_project_id/user_id/project_id

        :param user_id: (str) Id of the user.
        :param project_id: (str) Id of the project.
        '''
        self.user_id = user_id
        self.project_id = project_id


class Aspect_Ratio(db.Model):
    '''
    This class inherits from db.Model and is used to record the counts of
    each aspect_ratio in every projects.
    It is also used by ``SQLALCHEMY`` to store those relations records in the
    data base.
    '''
    aspect_ratio_id = db.Column(
        db.Integer, primary_key=True, unique=True, autoincrement=True)
    project_id = db.Column(
        db.Integer, ForeignKey(Project.project_id), nullable=False)
    aspect_ratio = db.Column(db.String(80), nullable=False)
    count = db.Column(db.Integer, nullable=False)

    def __init__(self, project_id, aspect_ratio, count):
        '''
        Set the attributes of for the relations between
        project_id/aspect_ratio/count
        
        :param project_id: (str) Id of the project.
        :param aspect_ratio: (float) ratio of width to heigh.
        :param count: (int) Count of each aspect ratio of each project.
        '''
        self.project_id = project_id
        self.aspect_ratio = aspect_ratio
        self.count = count


class Pred_Results(db.Model):
    '''
    This class inherits from db.Model and is used to record the prediction
    results of images.
    It is also used by ``SQLALCHEMY`` to store those relations records in the
    data base.
    '''
    pred_results_id = db.Column(
        db.Integer, primary_key=True, unique=True, autoincrement=True)
    project_id = db.Column(
        db.Integer, ForeignKey(Project.project_id), nullable=False)
    path_to_img = db.Column(db.String(200), nullable=False)
    label = db.Column(db.String(80), nullable=False)

    def __init__(self, project_id, path_to_img, label):
        '''
        Set the attributes of for the relations between
        project_id/aspect_ratio/count

        :param project_id: (str) Id of the project.
        :param path_to_img: (str) url path to the image.
        :param label: (str) prediction result.
        '''
        self.project_id = project_id
        self.path_to_img = path_to_img
        self.label = label


db.create_all()
db.session.commit()


@login_manager.user_loader
def load_user(id):
    '''
    Load the user record associated with `id` from the User table.

    :param id: (str) id of the user.
    :return: (User) User record (class instance)
    '''
    return User.query.get(int(id))
