import os
from app import application, classes, db

from flask import (render_template,
                   redirect,
                   url_for,
                   request,
                   flash,
                   jsonify)

import json
import boto3
from flask_wtf.file import FileField, FileRequired
import matplotlib.image as mpimg
import tempfile
from wtforms import SubmitField
from werkzeug import secure_filename
import shutil
import ml
import io

# for building forms
from flask_wtf import FlaskForm  # RecaptchaField
from wtforms import SubmitField, MultipleFileField
from wtforms import TextField, PasswordField
from wtforms.validators import DataRequired, Email, Length

# for handling log-in, log-out states
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash
from flask_login import current_user, login_user, login_required, logout_user

# for upload to s3
from boto.s3.key import Key
import boto
from collections import Counter
import time
from threading import Thread 
# for prediction
import numpy as np
from ml import train_ml, predict_ml, send_notification
# for download
import shutil
from flask import send_file


CLIENT = boto3.client('s3', aws_access_key_id='',
                      aws_secret_access_key='')

BUCKET_NAME = 'msds603-deep-vision'
MIN_IMG_LBL = 5

# Web app backend ##############

# S3 helper functions
def get_deepVision_bucket():
    bucket_name = 'msds603-deep-vision'
    s3_connection = boto.connect_s3(
        aws_access_key_id=' ',
        aws_secret_access_key=' ')
    # to be replaced: os.environ['AWS_ACCESS_KEY_ID'], ['AWS_SECRET_ACCESS_KEY']
    return s3_connection.get_bucket(bucket_name)


@application.route('/home')
@application.route('/index')
@application.route('/')
def index():
    """
    Route to the home page which can be accessed at / or /index or /home.
    """
    return render_template("index.html")


@application.route('/blog')
def blog():
    """Route to the blog page."""
    return render_template("blog.html")


@application.route('/blog-details')
def blog_details():
    """Route to the blog details page."""
    return render_template("blog-details.html")


@application.route('/contact')
def contact():
    """Route to the statis page about contact information."""
    return render_template("contact.html")


@application.route('/feature')
def feature():
    """Route to the statis page about service information."""
    return render_template("feature.html")


@application.route('/pricing')
def pricing():
    """Route to the statis page listing pricing information."""
    return render_template("pricing.html")


@application.route('/register', methods=['GET', 'POST'])
def register():
    """
    This function uses method request to take user-input data from a regular
    html form (not a FlaskForm object) then inserts the information of a
    new user into the database using SQLAlchemy.
    If data is valid, dedirect to log in page.
    Oherwise, render the sign up form again.
    """
    if request.method == "POST":
        username = request.form['username']
        companyname = request.form['companyname']
        email = request.form['email']
        password = request.form['password']

        user_count = classes.User.query.filter_by(username=username).count() \
            + classes.User.query.filter_by(email=email).count()

        if user_count == 0:
            user = classes.User(username, email, companyname, password)
            db.session.add(user)
            db.session.commit()
            # flash('successfully logged in.')
            return redirect(url_for('signin'))
        else:
            flash('Username or email already exists.')

    return render_template("signup.html")


@application.route('/signin', methods=['GET', 'POST'])
def signin():
    """
    This function uses method request to take user-input data from a regular
    html form (not a FlaskForm object) then queries user information in the
    database to log user in.
    If user information is found, redirect the user to project page.
    Otherwise, render the sign in form again.
    """
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = classes.User.query.filter_by(username=username).first()
        if user is not None and user.check_password(password):
            login_user(user)
            return redirect(url_for('projects'))
        else:
            flash('Invalid username and password combination.')

    return render_template("signin.html")


@application.route('/projects', methods=['GET', 'POST'])
@login_required
def projects():
    """
    This route displays the projects of a given user
    and allows them the ability to add a project.
    If a project using the same project_name already exists,
    this will display an error to tell the user
    to pick another project name.
    Project details are listed in a table, allowing user to
    upload image data per project, per label, to initiate
    training process and to upload new image for prediction.
    """
    if request.method == 'GET':
        users_projects = classes.User_Project.query.filter_by(user_id=current_user.id).all()
        project_ids = [user_project.project_id for user_project in users_projects]
        projects = classes.Project.query.filter(classes.Project.project_id.in_(project_ids))

        # proj_owners = {}
        # for proj in projects:
        #     proj_owners[proj.project_id] = classes.User.query.filter_by(id=proj.project_owner_id).first().username

        # return objects to easily call by attribute names,
        # rather than by indexes, please keep
        proj_labs = {}
        for proj in projects:
            proj_labs[proj.project_id] = classes.Label.query.filter_by(
                project_id=proj.project_id).all()
        # use dictionary to easily call by key which matches the ids,
        # more secure than by indexes, please keep

        return render_template('projects.html', projects=projects,
                               proj_labs=proj_labs
                               # , proj_owners=proj_owners
                               )

    elif request.method == 'POST':
        project_name = request.form['project_name']
        labels = [label.strip() for label in request.form['labels'].split(',')]

        # this was updated
        if len(set(labels)) != len(labels):
            return f"<h1>There are duplicate labels. Please enter labels that are different.</h1>"
        # TODO: verify label_names to be unique within one project,
        # TODO: right now can have same name but different labelid.

        # query the Project table to see if the project already exists
        # if it does, tell the user to pick another project_name
        users_projects = classes.User_Project.query.filter_by(user_id=current_user.id).all()
        project_ids = [user_project.project_id for user_project in users_projects]

        # if a user has multiple projects
        # check if the project name with the same name already exists for them
        projects_with_same_name = []
        if len(users_projects) > 0:
            projects = classes.Project.query.filter_by(project_name=project_name).all()
            projects_with_same_name = [project.project_id for project in projects if project.project_id in project_ids]

        if len(projects_with_same_name) > 0:
            return f"<h1> A project with the name: {project_name}" + \
                   " already exists. Please choose another " \
                   "name for your project.</h1>"
            # flash("A project with the same name already exists."
            #       "Please choose another name.")
            # return url_for('projects')

        else:
            # insert into the Project table
            db.session.add(classes.Project(project_name, int(current_user.id)))

            # get the project for the current user that was just added
            # (by using the creation date)
            most_recent_project = classes.Project.query \
                .filter_by(project_owner_id=current_user.id) \
                .order_by(classes.Project.project_creation_date.desc()).first()
            print(most_recent_project.project_name)

            # insert into the User_Project table
            # so that the user is associated with a project
            db.session.add(classes.User_Project(int(current_user.id),
                                                most_recent_project.project_id))

            # TODO: find a way to bulk insert
            # insert all of the labels that the user entered
            for label_idx, label in enumerate(labels):
                db.session.add(classes.Label(most_recent_project.project_id,
                                             label, label_idx))

            most_recent_project_labels = classes.Label.query.filter_by(project_id=most_recent_project.project_id)

            # this was added
            # TODO: when creating the project, I need to create the model 
            # and prediction folders for a given project in S3 

            bucket = get_deepVision_bucket()
            # bucket.set_acl('public-read')
            k = Key(bucket)

            for label in most_recent_project_labels:
                k.key = f'/{str(most_recent_project.project_id)}/{str(label.label_id)}/'
                k.set_contents_from_string('')

            k.key = f'/{str(most_recent_project.project_id)}/model/'
            k.set_contents_from_string('')

            k.key = f'/{str(most_recent_project.project_id)}/prediction/'
            k.set_contents_from_string('')


            k = Key(bucket)
            k.key = f'/{str(most_recent_project.project_id)}/'
            k.set_contents_from_string('')

            # pass the list of projects (including the new project) to the page
            # so it can be shown to the user
            # only commit the transactions once everything has been entered.
            db.session.commit()
            
            users_projects = classes.User_Project.query.filter_by(user_id=current_user.id).all()
            project_ids = [user_project.project_id for user_project in users_projects]
            projects = classes.Project.query.filter(classes.Project.project_id.in_(project_ids))
            
            proj_owners = {}
            for proj in projects:
                proj_owners[proj.project_id] = classes.User.query.filter_by(id=proj.project_owner_id).first().username
            
            proj_labs = {}
            for proj in projects:
                proj_labs[proj.project_id] = classes.Label.query.filter_by(
                    project_id=proj.project_id).all()

            return render_template('projects.html', projects=projects,
                                   proj_labs=proj_labs
                                   # , proj_owners=proj_owners
                                   )


class UploadFileForm(FlaskForm):
    """Class for uploading multiple files when submitted"""
    file_selector = MultipleFileField('File')
    submit = SubmitField('Submit')


@application.route('/upload/<labid>', methods=['GET', 'POST'])
@login_required
def upload(labid):
    """
    This route allows users to bulk upload image data per project, per label.
    Files would be stored in S3 bucket organized as "./project/label/files".
    """
    accepts = ['bmp', 'dib','jpeg', 'jpg', 'jpe', 'jp2', 'png', 'webp',
               'pbm', 'pgm', 'ppm', 'sr', 'ras','tiff', 'tif']

    label = classes.Label.query.filter_by(label_id=labid).first()
    labelnm = label.label_name
    projid = label.project_id

    projnm = classes.Project.query.filter_by(project_id=projid) \
        .first().project_name

    form = UploadFileForm()
    nfiles = 0
    muploaded = 0
    if form.validate_on_submit():
        files = form.file_selector.data
        nfiles = len(files)
        aspect_ratios_newfs = []  # aspect ratios of the each image
        for f in files:
            if f.filename.split('.')[-1].strip().lower() not in accepts:
                continue  # filter out those
            tmp = tempfile.NamedTemporaryFile()
            file_content = ''
            # file must be temporarily created so that it can be read
            # to find out the aspect ratio of the image
            with open(tmp.name, 'wb') as data:
                file_content = f.stream.read()
                data.write(file_content)
            # TODO: rename the images to be unique per project
            # and based on the count of images already stored
            # so that if you upload the same image multiple times
            # there will be multiple copies with different filenames
            # rather than updating the same file
            image = mpimg.imread(tmp.name)
            aspect_ratio = round(float(image.shape[1])/float(image.shape[0]), 1)
            filename = secure_filename(f.filename)

            aspect_ratios_newfs.append(str(aspect_ratio))

            # send file to s3 one by one
            bucket = get_deepVision_bucket()
            k = Key(bucket)
            ts = round(time.time())
            k.key = '/'.join([str(projid), str(labid), str(ts)+"_"+filename])
            k.set_contents_from_string(file_content)

        muploaded = len(aspect_ratios_newfs)
        # update the ratio tabel outside the loop over each file
        aspect_ratio_newcounts = Counter(aspect_ratios_newfs)
        # a dictionary of ratios: counts of new images
        for r in aspect_ratio_newcounts:
            aspect_ratios = classes.Aspect_Ratio.query\
                .filter_by(project_id=projid).filter_by(aspect_ratio=r).all()

            if len(aspect_ratios) == 0:
                db.session.add(classes.Aspect_Ratio(
                    projid, r, aspect_ratio_newcounts[r]))
            elif len(aspect_ratios) == 1:
                aspect_ratios[0].count += aspect_ratio_newcounts[r]

        db.session.commit()

    return render_template('upload_lab.html', projnm=projnm,labelnm=labelnm,
                           form=form, nfiles=nfiles, muploaded=muploaded)


@application.route('/train/<projid>', methods=['GET', 'POST'])
@login_required
def train(projid):
    """
    This route triggered when a user clicks "Train" button in a project.
    After training is done, the user will receive an notification email.
    """
    error = None
    # query inputs for to train the model

    # Check that minimum amount of images are uploaded
    # TODO: Frontend handeling this
    folders = CLIENT.list_objects(Bucket=BUCKET_NAME, Prefix=f'{projid}/', Delimiter="/")

    # if 'CommonPrefixes' in folders:
    dirs = [di['Prefix'] for di in folders['CommonPrefixes'] if
            di['Prefix'] not in [f'{projid}/prediction/', f'{projid}/model/']]

    if len(dirs)>0:
        for di in dirs:
            files = CLIENT.list_objects(Bucket=BUCKET_NAME, Prefix=di, Delimiter="")

            if 'Contents' in files:
                files = [f['Key'] for f in files['Contents'][1:]]
                if len(files) < MIN_IMG_LBL:
                    lbl_id = di.split('/')[1]
                    lbl_name = classes.Label.query.filter_by(project_id=projid, label_id=lbl_id).first().label_name
                    message = (f"You currently have uploaded {len(files)}, you are missing" +
                               f" {MIN_IMG_LBL - len(files)} to reach the minimum amount of images ({MIN_IMG_LBL}) for label {lbl_name}")
                    return redirect(url_for('projects'), message=message)

            else:
                lbl_id = di.split('/')[1]
                lbl_name = classes.Label.query.filter_by(project_id=projid, label_id=lbl_id).first().label_name
                message = f"Upload {MIN_IMG_LBL} images for {lbl_name} before starting training"
                return redirect(url_for('projects'), message=message)

    else:
        message = f"Upload {MIN_IMG_LBL} images for each label before start training"
        return redirect(url_for('projects'), message=message)


    print('Enters training route')
    max_asp_ratio = float(classes.Aspect_Ratio.query.filter_by(project_id=projid).order_by(classes.Aspect_Ratio.count.desc()).first().aspect_ratio)

    # proj_name = proj.project_name


    # last_asp_ratio = proj.last_train_asp_ratio
    
    project_owner_id = classes.Project.query.filter_by(project_id=projid).first().project_owner_id
    proj_owner = classes.User.query.filter_by(id=project_owner_id).first() 
    proj_owner_name = proj_owner.username
    proj_owner_email = proj_owner.email

    labels = classes.Label.query.filter_by(project_id=projid).all()
    print(labels)
    lbl2idx = {str(label.label_id): (label.label_index if label.label_index else i) for i, label in enumerate(labels)}

    # call the train function from ml module
    print('before training', lbl2idx)
    try:
        t = Thread(target=train_ml, args=(projid, max_asp_ratio, proj_owner_name, proj_owner_email, lbl2idx))
        t.start()
        message = "Your model is training now. You will receive an email once it's reday."
    except: message = "There has been an error training your model. Please contact our support email at info.deep.vision.co@gmail.com"
    #train_ml(projid, max_asp_ratio, proj_owner_name, proj_owner_email, lbl2idx)
    print("Model initiated in another thread.")
    return redirect(url_for('projects'), message=message)


@application.route('/training/<projid>', methods=['GET', 'POST'])
@login_required
def training(projid):
    """
    This route triggered when a user clicks "Train" button in a project.
    After training is done, the user will receive an notification email.
    """
    projnm = classes.Project.query.filter_by(project_id=projid) \
            .first().project_name

    error = None
    # query inputs for to train the model

    # Check that minimum amount of images are uploaded
    # TODO: Frontend handeling this
    folders = CLIENT.list_objects(Bucket=BUCKET_NAME, Prefix=f'{projid}/', Delimiter="/")

    # if 'CommonPrefixes' in folders:
    dirs = [di['Prefix'] for di in folders['CommonPrefixes'] if
            di['Prefix'] not in [f'{projid}/prediction/', f'{projid}/model/']]
    label_with_img_num = []
    # CALCULATE THE NUM OF IMGS FOR EACH LABEL
    all_labels = classes.Label.query.filter_by(project_id=projid).all()
    all_labels = [lb.label_name for lb in all_labels]
    
    if request.method=='GET':
        if len(dirs)>0:
            for di in dirs:
                files = CLIENT.list_objects(Bucket=BUCKET_NAME, Prefix=di, Delimiter="")
                lbl_id = di.split('/')[1]
                lbl_name = classes.Label.query.filter_by(project_id=projid, label_id=lbl_id).first().label_name

                if 'Contents' in files:
                    files = [f['Key'] for f in files['Contents'][1:]]
                    num_files_with_lbl = len(files)
                    if num_files_with_lbl < MIN_IMG_LBL:
                        # lbl_id = di.split('/')[1]
                        # lbl_name = classes.Label.query.filter_by(project_id=projid, label_id=lbl_id).first().label_name
                        message = (f"You currently have uploaded {len(files)}, you are missing" +
                                f" {MIN_IMG_LBL - len(files)} to reach the minimum amount of images ({MIN_IMG_LBL}) for label {lbl_name}")
                        # return redirect(url_for('projects'), message=message)
                        label_with_img_num.append((lbl_name, num_files_with_lbl))
                        for lb in all_labels:
                            if lb not in [i[0] for i in label_with_img_num]:
                                label_with_img_num.append((lb, 0))
                        return render_template('training.html', projnm=projnm, message=message, projid=projid, \
                            label_with_img_num=label_with_img_num)
                    else:
                        label_with_img_num.append((lbl_name, num_files_with_lbl))
                else:
                    # lbl_id = di.split('/')[1]
                    # lbl_name = classes.Label.query.filter_by(project_id=projid, label_id=lbl_id).first().label_name
                    message = f"Upload {MIN_IMG_LBL} images for {lbl_name} before starting training"
                    num_files_with_lbl = 0
                    label_with_img_num.append((lbl_name, num_files_with_lbl))
                    for lb in all_labels:
                        if lb not in [i[0] for i in label_with_img_num]:
                            label_with_img_num.append((lb, 0))
                    return render_template('training.html', projnm=projnm,
                                           message=message, projid=projid,
                                           label_with_img_num=label_with_img_num)

        else:
            message = f"Upload {MIN_IMG_LBL} images for" \
                      f"each label before start training"
            # return redirect(url_for('projects'), message=message)
            for lb in all_labels:
                if lb not in [i[0] for i in label_with_img_num]:
                    label_with_img_num.append((lb, 0))
            return render_template('training.html', projnm=projnm,
                                   message=message, projid=projid,
                                   label_with_img_num=label_with_img_num)

        print('Enters training route')
        max_asp_ratio = float(classes.Aspect_Ratio.query
                              .filter_by(project_id=projid)
                              .order_by(classes.Aspect_Ratio.count.desc())
                              .first().aspect_ratio)

        # proj_name = proj.project_name
        # last_asp_ratio = proj.last_train_asp_ratio
        project_owner_id = classes.Project.query\
                                  .filter_by(project_id=projid)\
                                  .first().project_owner_id
        proj_owner = classes.User.query.filter_by(id=project_owner_id).first()
        proj_owner_name = proj_owner.username
        proj_owner_email = proj_owner.email

        labels = classes.Label.query.filter_by(project_id=projid).all()
        print(labels)
        lbl2idx = {str(label.label_id): (label.label_index
                                         if label.label_index else i)
                   for i, label in enumerate(labels)}

        # call the train function from ml module
        print('before training', lbl2idx)
        try:
            t = Thread(target=train_ml, args=(projid, max_asp_ratio,
                                              proj_owner_name,
                                              proj_owner_email, lbl2idx))
            t.start()
            message = "Your model is training now. " \
                      "You will receive an email once it's ready."
        except:
            message = "There has been an error training your model. " \
                      "Please contact our support email at " \
                      "info.deep.vision.co@gmail.com"
        # train_ml(projid, max_asp_ratio, proj_owner_name,
        # proj_owner_email, lbl2idx)
        print("Model initiated in another thread.")
        # return redirect(url_for('projects'), message=message)

        return render_template('training.html', projnm=projnm, message=message,
                               projid=projid,
                               label_with_img_num=label_with_img_num)


@application.route('/predict/<projid>', methods=['GET', 'POST'])
@login_required
def predict(projid):
    """
    This route provides prediction on newly uploaded image for a project.
    :return: predicted label of the new image, display on the website.
    """

    client = boto3.client('s3',
                          aws_access_key_id=' ',
                          aws_secret_access_key=' ')
    bucket_name = 'msds603-deep-vision'

    projnm = classes.Project.query.filter_by(project_id=projid) \
        .first().project_name

    # check if there is a model
    filepaths = client.list_objects(Bucket=bucket_name,
                                    Prefix=projid, Delimiter='')
    filepaths = [item['Key'] for item in filepaths['Contents']
                 if item['Key'].split('/')[0] == projid]
    if f'{projid}/model/' not in filepaths:
        return "A model has to be trained before predicting."

    form = UploadFileForm()
    project_predictions = classes.Pred_Results.query\
                                 .filter_by(project_id=projid).all()

    if form.validate_on_submit():
        files = form.file_selector.data
        accepts = ['bmp', 'dib', 'jpeg', 'jpg', 'jpe', 'jp2', 'png', 'webp',
                   'pbm', 'pgm', 'ppm', 'sr', 'ras', 'tiff', 'tif']

        # Get aspect_ratio
        project = classes.Project.query.filter_by(project_id=projid).first()
        aspect_ratio = project.last_train_asp_ratio

        # remove and create the tmp sub directory for that project
        if not os.path.exists("/home/ec2-user/"
                              "product-analytics-group-project-deepvision/"
                              "code/app/static/tmp"):
            os.mkdir("/home/ec2-user/"
                     "product-analytics-group-project-deepvision/"
                     "code/app/static/tmp/")
        if os.path.exists(f"/home/ec2-user/"
                          f"product-analytics-group-project-deepvision/"
                          f"code/app/static/tmp/{projid}"):
            shutil.rmtree(f"/home/ec2-user/"
                          f"product-analytics-group-project-deepvision/"
                          f"code/app/static/tmp/{projid}")
        os.mkdir(f"/home/ec2-user/"
                 f"product-analytics-group-project-deepvision/"
                 f"code/app/static/tmp/{projid}")

        # Store imgs to s3 and ec2.
        s3_filepaths = []
        ec2_filepaths = []
        print(files)
        for f in files:
            if hasattr(f, 'filename') and \
                    f.filename.split('.')[-1].strip().lower() not in accepts:
                # if f.split('.')[-1].strip().lower() not in accepts:
                continue
            ts = time.time()
            filename = str(round(ts)) + "_" + secure_filename(f.filename)
            s3_filepath = '/'.join([str(projid), 'prediction', filename])
            # ec2_filepath = '/'.join(['/home/ec2-user/'
            #                          'product-analytics-group-project-deepvision/'
            #                          'code/app/static/tmp',
            #                          str(projid), filename])
            ec2_filepath = f"/home/ec2-user/" \
                           f"product-analytics-group-project-deepvision/" \
                           f"code/app/static/tmp/{projid}/{filename}"

            file_content = f.stream.read()

            client.put_object(Body=file_content, Bucket=BUCKET_NAME,
                              Key=s3_filepath, ACL='public-read')

            s3_filepaths.append(s3_filepath)
            ec2_filepaths.append(ec2_filepath)

            with open(ec2_filepath, 'wb') as data:
                file_content = f.stream.read()
                data.write(file_content)

        labels = classes.Label.query.filter_by(project_id=projid).all()
        model = ml.list_items(client, path=f"{projid}/model/",
                              only_folders=False, bucket_name=bucket_name)
        filepaths = [item['Key'] for item in model['Contents']
                     if len(item['Key'].split('.')) > 1]

        # if model exists
        if len(filepaths) == 0:
            flash("Please train a model before trying to predict.")
            render_template('predict.html', projnm=projnm,
                            project_predictions=project_predictions,
                            form=form, projid=projid)
        else:
            if aspect_ratio is None:
                aspect_ratio = 1
            predictions = predict_ml(project_id=projid, paths=s3_filepaths,
                                     aspect_r=aspect_ratio,
                                     n_training_labels=len(labels))
            idx2lbls = [0]*len(labels)
            for label in labels:
                idx2lbls[int(label.label_index)] = label.label_name
            prediction_labels = [idx2lbls[p] for p in predictions]
            project_ids = [projid for i in range(len(prediction_labels))]

            for row in zip(project_ids, s3_filepaths, prediction_labels):
                prediction(row[0], row[1], row[2])

            project_predictions = classes.Pred_Results.query\
                                         .filter_by(project_id=projid).all()

    return render_template('predict.html', projnm=projnm,
                           project_predictions=project_predictions,
                           form=form, projid=projid)


@application.route('/status/<projid>', methods=['GET', 'POST'])
@login_required
def status(projid):
    """
    This route provides project status,
    including the prediction results and users of this project.
    At this route, a user can also add other users to their project,
    one user at a time.
    Added users will also receive an email notifying them that they are
    added.
    """
    proj = classes.Project.query.filter_by(project_id=projid) \
        .first()
    project_predictions = classes.Pred_Results.query\
                                 .filter_by(project_id=projid).all()

    for project in project_predictions:
        print(project.path_to_img)
    projnm = proj.project_name
    proj_owner = classes.User.query\
                        .filter_by(id=proj.project_owner_id).first().username

    userids_of_one_project = classes.User_Project.query \
        .filter_by(project_id=projid).all()
    users = []
    for user_proj in userids_of_one_project:
        users.append(classes.User.query.filter_by(
            id=user_proj.user_id).first().username)

    # add user functionality
    if request.method == "POST":
        added_username = request.form['username']
        if len(added_username.split()) > 1:
            flash("You can only add one user at a time.")
            render_template('status.html', projnm=projnm,
                            users=users, projid=projid)

        found_users = classes.User.query\
                             .filter_by(username=added_username).all()

        if len(found_users) == 0:
            flash('User does not exist.')
            render_template('status.html', projnm=projnm,
                            users=users, projid=projid)

        elif added_username in users:
            flash(added_username + ' already has access to the project.')
            render_template('status.html', projnm=projnm,
                            users=users, projid=projid)

        else:
            user = classes.User.query\
                          .filter_by(username=added_username).first()
            user_id = user.id
            user_proj = classes.User_Project(user_id, projid)
            db.session.add(user_proj)
            db.session.commit()

            # send notification to the added user
            user_email = user.email
            subject_line = f"You have been added to project {projnm}."
            email_body = f"You have just been added to project {projnm}. " \
                         f"Now you can start uploading data, training, " \
                         f"and predicting."

            send_notification(added_username, user_email,
                              subject_line, email_body)

            users_with_new = []
            userids_with_new_of_one_project = classes.User_Project\
                .query.filter_by(project_id=projid).all()
            for user_proj_new in userids_with_new_of_one_project:
                users_with_new.append(
                    classes.User.query.filter_by(
                        id=user_proj_new.user_id).first().username
                )

            return render_template('status.html', projnm=projnm,
                                   proj_owner=proj_owner,
                                   users=users_with_new, projid=projid)
    return render_template('status.html', projnm=projnm, proj_owner=proj_owner,
                           users=users, projid=projid,
                           project_predictions=project_predictions)


@application.route('/downloadpred/<projid>', methods=['GET', 'POST'])
@login_required
def downloadpred(projid):
    """
    rename the files in tmp folder of the project
    zip all the files, send to user
    :param projid:
    :return: send files
    """
    preds = classes.Pred_Results.query.filter_by(project_id=projid).all()
    prediction_images = []
    filenames = []
    for p in preds:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        filenames.append(tmp.name)
        with open('/home/ec2-user/product-analytics-group-project-deepvision/'
                  'code/prediction' + tmp.name, 'wb') as data:
            CLIENT.download_fileobj(BUCKET_NAME, p.path_to_img, data)
    # prediction_images.append(img)
    # newnm = p.path_to_img.split('.')[0] + "_" + p.label + "_" +
    #         p.path_to_img.split('.')[1]
    # os.rename(p.path_to_img, newnm)
        # suppose the path in db is absolute
    projfolder = f'{projid}/prediction/'
    if len(preds) > 0:
        shutil.make_archive('prediction', 'zip',
                            '/home/ec2-user/'
                            'product-analytics-group-project-deepvision/'
                            'code/app/prediction')
        # shutil.make_archive(projfolder, 'zip', projfolder)
    try:
        return send_file('/home/ec2-user/'
                         'product-analytics-group-project-deepvision/'
                         'code/app/prediction.zip',
                         as_attachment=True,
                         attachment_filename='prediction.zip')
    except Exception as e:
        return str(e)


@application.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@application.errorhandler(401)
def unauthorized(e):
    """If user goes to a page that requires authorization such as
    projects page but is not yet logged in, redirect them to
    the log-in page."""
    return redirect(url_for('signin'))


# Mobile app backend ##############


@application.route('/mobile_register', methods=['GET', 'POST'])
def mobile_register():
    """
    This function uses method post to register user information
    in the data base. This returns a message indicating the if
    the procedure has been successful or not.

    :return: return dictionary with success "1"/"0" if success/failure
    """
    if request.method == "POST":
        username = request.form['username']
        companyname = request.form['companyname']
        email = request.form['email']
        password = request.form['password']

        user_count = classes.User.query.filter_by(username=username).count() \
            + classes.User.query.filter_by(email=email).count()

        if user_count == 0:
            user = classes.User(username, email, companyname, password)
            db.session.add(user)
            db.session.commit()
            # return "1" #
            return json.dumps({"success": "1"})

    # return "0"
    return json.dumps({"success": "0"})


@application.route('/mobile_signin', methods=['GET', 'POST'])
def mobile_signin():
    """
    Mobile version of signin
    This function uses method post to take user-input data and check if exists
    and the credentials are correct.

    :return: return dictionary with success "1"/"0" if success/failure
    """
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = classes.User.query.filter_by(email=email).first()

        if user is not None and user.check_password(password):
            login_user(user)
            # return "1"
            return json.dumps({"success": "1"})

    # return "0"
    return json.dumps({"success": "0"})


@application.route('/mobile_projects', methods=['GET', 'POST'])
@login_required
def mobile_projects():
    """
    This route is the backend for the project menu of a
    given user in the mobile phone.
    It allows for projects to be added.
    If a project using the same project_name already exists,
    this will display an error to tell the user
    to pick another project name.

    :return: return dictionary with success "1"/"0" if success/failure
    and the list of projects
    """

    if request.method == 'GET':
        projects = classes.User_Project.query.filter_by(
            user_id=int(current_user.id)).all()
        # return objects to easily call by attribute names,
        # rather than by indexes, please keep
        proj_labs = {}
        for proj in projects:
            proj_labs[proj.project_id] = classes.Label.query.filter_by(
                project_id=proj.project_id).all()
        # use dictionary to easily call by key which matches the ids,
        # more secure than by indexes, please keep

        return json.dumps(
            {"success": "1", "projects": json.dumps(projects),
             "proj_labs": json.dumps(proj_labs)})
    elif request.method == 'POST':
        project_name = request.form['project_name']
        labels = [label.strip() for label in request.form['labels'].split(',')]

        # query the Project table to see if the project already exists
        # if it does, tell the user to pick another project_name
        projects_with_same_name = classes.User_Project.query.filter_by(
            project_name=project_name).all()
        if len(projects_with_same_name) > 0:
            return json.dumps({"success": "0"})
        else:
            # insert into the Project table
            db.session.add(classes.Project(project_name, int(current_user.id)))

            # get the project for the current user that was just added
            # (by using the creation date)
            most_recent_project = classes.Project.query.filter_by(
                project_owner_id=current_user.id) \
                .order_by(classes.Project.project_creation_date.desc()).first()

            # insert into the User_Project table so that the
            # user is associated with a project
            db.session.add(classes.User_Project(int(current_user.id),
                                                most_recent_project.project_id,
                                                project_name))

            for label in labels:
                db.session.add(classes.Label(most_recent_project.project_id,
                                             label))

            # pass the list of projects (including the new project)
            # to the page so it can be shown to the user
            # only commit the transactions once everything
            # has been entered successfully.
            db.session.commit()

            projects = classes.User_Project.query.filter_by(user_id=int(
                current_user.id)).all()
            proj_labs = {}
            for proj in projects:
                proj_labs[proj.project_id] = classes.Label.query.filter_by(
                    project_id=proj.project_id).all()

            # return render_template('projects.html',
            # projects=projects, proj_labs=proj_labs)
            return json.dumps(
                {"success": "1", "projects": json.dumps(projects),
                 "proj_labs": json.dumps(proj_labs)})


@application.route('/mobile_logout')
@login_required
def mobile_logout():
    '''
    Log out function for mobile phone
    :return: return dictionary with success "1"/"0" if success/failure
    '''
    logout_user()
    # flash('You have been logged out.')
    # return "1"
    return json.dumps({"success": "1"})

# TODO: mobile_upload


def prediction(project_id, path_to_img, label):
    '''
    Record the predicted result and their path
    '''
    pred_results = classes.Pred_Results(project_id, path_to_img, label)
    db.session.add(pred_results)
    db.session.commit()
