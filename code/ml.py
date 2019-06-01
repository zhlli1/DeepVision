import asyncio
import ssl
import smtplib
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import boto3
import numpy as np
from functools import partial
import datetime
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import models
import pandas as pd
import matplotlib.image as mpimg
import tempfile
import cv2
import warnings
warnings.filterwarnings('ignore')

# Global variables
BUCKET_ORIG = 'msds603-deep-vision'
BUCKET_RESIZE = 'msds603-deep-vision-predict'
TMP_MODEL_FILE = '_tmp_model.pth'
TMP_IMG_FILE = '_tmp_img.jpg'
MODEL_W_FOLD_NAME = 'model'
PREDICTION_FOLDER_NAME = 'prediction'
CLIENT = boto3.client('s3', aws_access_key_id='AKIAIQRI4EE5ENXNW6LQ',
                      aws_secret_access_key='2gduLL4umVC9j7XXc' +
                                            '2L1N8DfUVQQKcFmnezTYF8O')

BATCH_SIZE = 8
R_PIX = 8
MAX_EPOCHS = 10

SMTP_SERVER = "smtp.gmail.com"
PORT = 587  # For starttls
SENDER_EMAIL = 'info.deep.vision.co@gmail.com'
PASSWORD = 'D33pVisi0nPa55word'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using device {device}')

# Functions dealing with S3 #


def get_label_names(folder_dic):
    return folder_dic['Prefix'].split('/')[-2]


def list_items(client, path=None, only_folders=False, bucket_name=BUCKET_ORIG):
    delimiter = "/" if only_folders else ""
    if path is None:
        raise ValueError("A path in the bucket is needed")
    else:
        return client.list_objects(Bucket=bucket_name,
                                   Prefix=path,
                                   Delimiter=delimiter)


def get_project_df(client, project, bucket_name=BUCKET_ORIG):
    all_file_paths = []
    all_labels = []
    labels = list_items(client, path=f"{project}/",
                        only_folders=True, bucket_name=bucket_name)
    for folder_dic in labels['CommonPrefixes']:
        dir_name = folder_dic['Prefix'].split('/')[-2]
        if dir_name not in [MODEL_W_FOLD_NAME, PREDICTION_FOLDER_NAME]:

            label = get_label_names(folder_dic)
            files_obj = list_items(
                client, path=f"{project}/{label}/", only_folders=False)
            files_paths = [file_obj['Key']
                           for file_obj in files_obj['Contents'][1:]]
            all_file_paths.extend(files_paths)
            all_labels.extend([label]*len(files_paths))

    return pd.DataFrame(zip(all_file_paths, all_labels),
                        columns=['path', 'label'])

# Functions for Resizing #


def im_read_resize(path, aspect_r, client, bucket_name=BUCKET_ORIG):
    im = get_image(client, path, show=False, bucket_name=bucket_name)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    w, r, _ = im.shape
    im = cv2.resize(im, (250, int(aspect_r*250)))  # (width, high)
    return im


def im_write(im, client, s3_path,
             bucket_name=BUCKET_RESIZE,
             tmp_p='resize'+TMP_IMG_FILE):
    cv2.imwrite(tmp_p, im)
    client.upload_file(tmp_p,
                       bucket_name,
                       Key=s3_path,
                       ExtraArgs={'ACL': 'bucket-owner-full-control'})
    os.remove(tmp_p)


def resize_images(df, aspect_r, client,
                  project_name, org_bucket=BUCKET_ORIG,
                  dest_bucket=BUCKET_RESIZE):
    for org_p in df.path:
        im = im_read_resize(org_p, aspect_r, client, org_bucket)
        safe_path = project_name+'_resize'+TMP_IMG_FILE
        im_write(im, client, s3_path=org_p,
                 bucket_name=dest_bucket, tmp_p=safe_path)


# Creating the DataSet #


def get_image(client, file_path, show=False, bucket_name=BUCKET_RESIZE):

    tmp = tempfile.NamedTemporaryFile(delete=False)
    with open(tmp.name, 'wb') as data:
        client.download_fileobj(bucket_name, file_path, data)
    #    await asyncio.sleep(10)
    img = mpimg.imread(tmp.name)

    return img


def create_mappings(x: pd.Series):
    labels = x.unique()
    lbl2idx = {label: idx for idx, label in enumerate(labels)}
    return lbl2idx, labels


class ProjectDS(Dataset):

    def __init__(self, df, lbl2idx, client, project_name,
                 bucket_name=BUCKET_RESIZE):
        self.bucket_name = bucket_name
        self.client = client

        self.paths = df.path.to_list()

        self.lbl2idx = lbl2idx
        self.labels = list(map(lambda x: self.lbl2idx[x], df.label))

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):

        x = get_image(self.client, self.paths[idx], show=False,
                      bucket_name=self.bucket_name)

        x = x.astype(np.float32)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) / 255

        return x, self.labels[idx]

# Data Augmenation: Transformations #


def normalize_imagenet(im):
    """Normalizes images with Imagenet stats."""
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im - imagenet_stats[0])/imagenet_stats[1]


def crop(im, r, c, target_r, target_c): return im[r:r+target_r, c:c+target_c]


class RandomCrop:
    """
    Returns a random crop

    Those are the arguments used when specifiying the transformations.

    Args:
        target_r:int Target Width
        target_c:int Target Hight

    """

    def __init__(self, r_pix):
        self.r_pix = r_pix

    def __call__(self, x, rand_r, rand_c):
        """To be called in  the transform
        """

        r, c, *_ = x.shape

        c_pix = round(self.r_pix*c/r)

        start_r = np.floor(2 * rand_r * self.r_pix).astype(int)
        start_c = np.floor(2 * rand_c * c_pix).astype(int)

        return crop(x, start_r, start_c, r-2*self.r_pix, c-2*c_pix)

    def options(self, x_shape):
        """Specify the random arguments to be generated every epoch.
        Images must be have same dimensions !
        """
        r, c, *_ = x_shape
        return {"rand_r": -1,
                "rand_c": -1}

    def set_random_choices(self, N, x_shape):
        return {k: (v*np.random.uniform(0, 1, size=N)).astype(int)
                for k, v in self.options(x_shape).items()}


class RandomRotation:
    """ Rotates an image by deg degrees

    Args: -
    """

    def __init__(self, arc_width: float = 20): self.arc_width = arc_width

    def __call__(self, im, deg,
                 mode=cv2.BORDER_REFLECT, interpolation=cv2.INTER_AREA):
        r, c, *_ = im.shape
        M = cv2.getRotationMatrix2D((c / 2, r / 2), deg, 1)
        return cv2.warpAffine(im, M, (c, r), borderMode=mode,
                              flags=cv2.WARP_FILL_OUTLIERS + interpolation)

    def options(self, x_shape):
        """Specify the random arguments to be generated every epoch.
        Images must be have same dimensions !
        """
        return {"deg": -1}

    def set_random_choices(self, N, x_shape):
        return {k: ((np.random.random(size=N) - .50)*self.arc_width)
                for k, v in self.options(x_shape).items()}


class Flip:
    """ Rotates an image by deg degrees
    Args: -
    """

    def __init__(self): pass

    def __call__(self, im, flip):
        if flip > .5:
            im = np.fliplr(im).copy()
        return im

    def options(self, x_shape):
        """Specify the random arguments to be generated every epoch.
        Images must be have same dimensions !
        """
        return {"flip": -1}

    def set_random_choices(self, N, x_shape):
        return {k: np.random.random(size=N)
                for k, v in self.options(x_shape).items()}


def center_crop(im, r_pix=8):
    """ Returns a center crop of an image"""
    r, c, *_ = im.shape
    c_pix = round(r_pix*c/r)
    return crop(im, r_pix, c_pix, r-2*r_pix, c-2*c_pix)


# Data Augmentation: Wrapper for Transformations #

class Transform():

    def __init__(self, dataset, transforms=None, normalize=True, r_pix=8):
        self.dataset, self.transforms = dataset, transforms

        if normalize is True:
            self.normalize = normalize_imagenet
        else:
            self.normalize = False

        self.center_crop = partial(center_crop, r_pix=r_pix)

    def __len__(self): return len(self.dataset)

    def __getitem__(self, index):

        data, label = self.dataset[index]

        if self.transforms:
            for choices, f in list(zip(self.choices, self.transforms)):
                args = {k: v[index] for k, v in choices.items()}
                data = f(data, **args)
        else:
            data = self.center_crop(im=data)

        if self.normalize:
            data = self.normalize(data)

        return np.rollaxis(data, 2), label

    def set_random_choices(self):
        """
        To be called at the begining of every epoch
        to generate the random numbers
        for all iterations and transformations.
        """
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)

        for t in self.transforms:
            self.choices.append(t.set_random_choices(N, x_shape))


# Model Storing/Loading Functions #


def now_str(): return datetime.datetime.strftime(
    datetime.datetime.now(), "%Y-%m-%d-%H-%M-%S")


def save_model(m, client, project_name,
               bucket_name=BUCKET_ORIG,
               tmp_p=TMP_MODEL_FILE, s3_p=None):

    safe_tmp_file = project_name+'_save'+tmp_p
    if s3_p is None:
        s3_p = f'{project_name}/{MODEL_W_FOLD_NAME}/' + now_str() + \
            '-model.pth'
    torch.save(m.state_dict(), safe_tmp_file)
    client.upload_file(safe_tmp_file, bucket_name, Key=s3_p,
                       ExtraArgs={'ACL': 'bucket-owner-full-control'})
    os.remove(safe_tmp_file)


def load_model(m, client, project_name,
               bucket_name=BUCKET_ORIG,
               tmp_p=TMP_MODEL_FILE):
    '''Loads the latest model for this project if exists'''
    model_p_objects = list_items(client,
                                 path=f"{project_name}/{MODEL_W_FOLD_NAME}/",
                                 only_folders=False, bucket_name=bucket_name)

    if 'Contents' in model_p_objects.keys():
        safe_tmp_file = project_name+'_load'+tmp_p
        model_ps = sorted([f['Key']
                           for f in model_p_objects['Contents']], reverse=True)
        path = model_ps[0]
        client.download_file(bucket_name, path, safe_tmp_file)
        m.load_state_dict(torch.load(safe_tmp_file, map_location='cpu'))
        os.remove(safe_tmp_file)

# Architecture #


class DenseNet(nn.Module):
    '''
    DenseNet121 with quick iterations on:
     > arbitrary finite out_size.
     > pre-trained model between ImageNet
     > and the medical image data-set MURA (all but last layer).
     > half float precision (16)
     > freeze layers
    '''
    def __init__(self, out_size: int = 1,
                 pretrained: bool = True,
                 freeze: str = False):
        '''

        :param out_size: (int) output size
        :param pretrained: (bool/str) Kind of pre-train: \
        Supports'MURA', True and False.
        :param freeze: (bool) freeze all layers but last one.
        '''
        super().__init__()

        top_model = models.densenet121(pretrained=pretrained)
        top_layers = list(top_model.children())[0]
        top_layers_groups = [top_layers[:7], top_layers[7:]]

        self.groups = nn.ModuleList(
            [nn.Sequential(*group) for group in top_layers_groups])
        self.groups.append(nn.Linear(1024, out_size))

        if freeze:
            self.freeze([0, 1])

    def forward(self, x):

        for group in self.groups[:-1]:
            x = group(x)
        x = F.relu(x)
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.shape[0], -1)
        x = self.groups[-1](x)
        return x

    def freeze(self, group_idxs: int):
        if not isinstance(group_idxs, (list, tuple)):
            group_idxs = [group_idxs]
        for group_idx in group_idxs:
            group = self.groups[group_idx]
            parameters = filter(lambda x: x.requires_grad, group.parameters())
            for p in parameters:
                p.requires_grad = False

    def unfreeze(self, group_idx: int):
        if group_idx not in [0, 1, 2]:
            raise ValueError('group_idx must be between 0 and 2')
        group = self.groups[group_idx]
        parameters = filter(lambda x: hasattr(
            x, 'requires_grad'), group.parameters())
        for p in parameters:
            p.requires_grad = True

# TRAINING FUNCTIONS #

# Validation functions


def ave_f1(probs, ys, n_lbls):
    #     ys2 = label_binarize(ys, classes=list(range(n_lbls)))
    #     print(ys2)
    #     print(ys2)
    #     aucs = [roc_auc_score(ys, probs) for i in range(probs.shape[1])]
    # np.mean(aucs)
    return f1_score(ys, np.argmax(probs, axis=1), average='weighted')


def validate_loop(model, valid_dl, task):

    if task == 'multiclass':
        loss_f = F.cross_entropy
    elif task == 'binary':
        loss_f = F.binary_cross_entropy_with_logits

    model.eval()
    total = 0
    sum_loss = 0
    ys = []
    preds = []

    for x, y in valid_dl:

        x, y = x.float().to(device), y.float().to(device)  # x.cuda(),y.cuda()
        out = model(x)
        loss = loss_f(out.squeeze(), y)

        batch = y.shape[0]
        sum_loss += batch * (loss.item())
        total += batch

        preds.append(out.detach().cpu().numpy())
        ys.append(y.long().cpu().numpy())

    preds = np.vstack(preds)
    ys = np.concatenate(ys)
    return sum_loss/total, preds, ys


def validate_multiclass(model, valid_dl, n_lbls):

    loss, preds, ys = validate_loop(model, valid_dl, 'multiclass')

    mean_auc = ave_f1(preds, ys, n_lbls)

    return loss, mean_auc


def validate_binary(model, valid_dl, n_lbls):

    loss, preds, ys = validate_loop(model, valid_dl, 'binary')

    auc = roc_auc_score(y_score=preds, y_true=ys)

    return loss, auc

# Policies


def cos_segment(start_lr, end_lr, n_iterations):
    i = np.arange(n_iterations)
    c_i = 1 + np.cos(i * np.pi / n_iterations)
    return end_lr + (start_lr - end_lr) / 2 * c_i


class TrainingPolicy:
    '''Cretes the lr and momentum policy'''

    def __init__(self, n_epochs, dl, max_lr,
                 pctg=.3, moms=(.95, .85),
                 delta=1e-4, div_factor=25.):

        total_iterations = n_epochs * len(dl)

        iter1 = int(total_iterations * pctg)
        iter2 = total_iterations - int(total_iterations * pctg)
        iterations = (iter1, iter2)

        min_start = max_lr/div_factor
        min_end = min_start*delta

        lr_segments = ((min_start, max_lr), (max_lr, min_end))
        mom_segments = (moms, (moms[1], moms[0]))

        self.lr_schedule = self._create_schedule(lr_segments, iterations)
        self.mom_schedule = self._create_schedule(mom_segments, iterations)

        self.idx = -1

    def _create_schedule(self, segments, iterations):
        '''
        Creates a schedule given a function, behaviour and size
        '''
        stages = [cos_segment(start, end, n)
                  for ((start, end), n) in zip(segments, iterations)]
        return np.concatenate(stages)

    def step(self):
        self.idx += 1
        return self.lr_schedule[self.idx], self.mom_schedule[self.idx]

# Optimizer and diff lr


def diff_range(val, alpha=1./3):
    return [val * alpha**(2-i) for i in range(3)]


class OptimizerWrapper:
    '''Without using the momentum policy'''

    def __init__(self, policy, model, n_epochs,
                 dl, max_lr, wd=0, alpha=1. / 3):

        self.policy = policy
        self.model = model
        self.wd = wd
        self.alpha = alpha

        # This assumes the model is defined by groups.
        param_groups = [group.parameters()
                        for group in list(self.model.children())[0]]

        lr_0 = self.policy.lr_schedule[0]
        mom_0 = self.policy.mom_schedule[0]

        # crete PyTorch optimizer with paramters groups
        self.optimizer = optim.Adam(
            [{'params': p, 'lr': lr, 'mom': (mom, .999)}
             for p, lr, mom in zip(param_groups,
                                   diff_range(lr_0, alpha),
                                   diff_range(mom_0, 1))]
        )

    def _update_optimizer(self):
        '''Updates the optimizer's lr and mom for next step'''
        lr_i, mom_i = self.policy.step()
        groups = zip(self.optimizer.param_groups, diff_range(
            lr_i, self.alpha), diff_range(mom_i, 1))
        for param_group, lr, mom in groups:
            param_group['lr'] = lr
            param_group['mom'] = (mom, .999)

    def _weight_decay(self):
        '''Does weight decay (properly) in the parameters'''
        for group in self.optimizer.param_groups:
            for p in group['params']:
                p.data.mul_(group['lr'] * self.wd)

    def step(self):
        '''Step to be called at the beginning of the iteration'''
        self._update_optimizer()
        if self.wd != 0:
            self._weight_decay()
        self.optimizer.step()

    def zero_grad(self): self.optimizer.zero_grad()

# training function


def train_model(n_epochs, model, train_dl, n_lbls,
                best_loss, valid_dl=None,
                max_lr=.01, wd=0, alpha=1. / 3,
                project_name=None,
                save_path=None,
                unfreeze_during_loop: tuple = None, client=CLIENT):
    model.train()
    if n_lbls > 2:
        validate = validate_multiclass
        loss_fun = F.cross_entropy
    else:
        validate = validate_binary
        loss_fun = F.binary_cross_entropy_with_logits

    if unfreeze_during_loop:
        total_iter = n_epochs*len(train_dl)
        first_unfreeze = int(total_iter*unfreeze_during_loop[0])
        second_unfreeze = int(total_iter*unfreeze_during_loop[1])

    cnt = 0

    policy = TrainingPolicy(n_epochs=n_epochs, dl=train_dl, max_lr=max_lr)
    optimizer = OptimizerWrapper(
        policy, model, n_epochs, train_dl, max_lr=max_lr, wd=wd, alpha=alpha)

    for epoch in range(n_epochs):
        model.train()
        train_dl.dataset.set_random_choices()
        agg_div = 0
        agg_loss = 0
        for x, y in train_dl:

            if unfreeze_during_loop:
                if cnt == first_unfreeze:
                    model.unfreeze(1)
                if cnt == second_unfreeze:
                    model.unfreeze(0)

            x, y = x.float().to(device), y.float().to(device)

            out = model(x)
            loss = loss_fun(input=out.squeeze(), target=y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch = y.shape[0]
            agg_loss += batch*loss.item()
            agg_div += batch
            cnt += 1

        val_loss, measure = validate(model, valid_dl, n_lbls=n_lbls)
        print(
            f'Ep. {epoch+1} - train loss {agg_loss/agg_div:.4f} ' +
            f'-  val loss {val_loss:.4f} AVG F1 {measure:.4f}')

        if val_loss < best_loss:
            save_model(model, client, project_name, bucket_name=BUCKET_ORIG,
                       tmp_p=TMP_MODEL_FILE, s3_p=save_path)

    return best_loss


# sending email

def send_notification(receiver_name, receiver_email, subject_line, email_body):
    """
    A general function to send notification emails to users.
    Subject line and content of the notification can be customized.
    """
    Text = f"""
    Hello {receiver_name.title()},

    {email_body}

    Best,
    Deep Vision Team"""

    message = f'Subject: {subject_line}\n\n{Text}'

    # Create a secure SSL context
    context = ssl.create_default_context()

    # Try to log in to server and send email
    try:
        server = smtplib.SMTP(SMTP_SERVER, PORT)
        server.ehlo()  # Can be omitted
        server.starttls(context=context)  # Secure the connection
        server.ehlo()  # Can be omitted
        server.login(SENDER_EMAIL, PASSWORD)
        server.sendmail(SENDER_EMAIL, receiver_email, message)
        # TODO: Send email here
    except Exception as e:
        # Print any error messages to stdout
        print(e)
    finally:
        server.quit()


def send_email(receiver_name, receiver_email):

    Subject = 'Your training has finished'
    Text = f"""Hello {receiver_name.title()},

    You training has finished. You can log-in and
     start using your brand new model.

    Best,
    Deep Vision Team
    """

    message = f'Subject: {Subject}\n\n{Text}'

    # Create a secure SSL context
    context = ssl.create_default_context()

    # Try to log in to server and send email
    try:
        server = smtplib.SMTP(SMTP_SERVER, PORT)
        server.ehlo()  # Can be omitted
        server.starttls(context=context)  # Secure the connection
        server.ehlo()  # Can be omitted
        server.login(SENDER_EMAIL, PASSWORD)
        server.sendmail(SENDER_EMAIL, receiver_email, message)
        # TODO: Send email here
    except Exception as e:
        # Print any error messages to stdout
        print(e)
    finally:
        server.quit()


# calls from backend #


def train_ml(proj_id, aspect_r, name, email, lbl2idx):
    # Resizing
    print('Resizing images ...')
    df = get_project_df(CLIENT, proj_id, BUCKET_ORIG)
    resize_images(df, aspect_r, CLIENT, proj_id)

    print('Creating data sets/ loaders ...')
    # datasets declaration
    transforms = [RandomRotation(arc_width=30), Flip(), RandomCrop(R_PIX)]
    df = get_project_df(CLIENT, proj_id, BUCKET_RESIZE)

    train_X, valid_X, train_Y, valid_Y = train_test_split(df.path.values,
                                                          df.label.values,
                                                          test_size=.1)
    train_df = pd.DataFrame({'path': train_X, 'label': train_Y})
    valid_df = pd.DataFrame({'path': valid_X, 'label': valid_Y})

    train_ds = Transform(ProjectDS(train_df, lbl2idx,
                                   CLIENT, proj_id, BUCKET_RESIZE),
                         transforms=transforms,
                         normalize=False, r_pix=R_PIX)
    valid_ds = Transform(ProjectDS(valid_df, lbl2idx,
                                   CLIENT, proj_id, BUCKET_RESIZE),
                         transforms=False,
                         normalize=False, r_pix=R_PIX)

    n_lbls = len(lbl2idx)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)

    save_path = f'{proj_id}/{MODEL_W_FOLD_NAME}/' + \
        now_str() + '-model.pth'

    best_loss = np.inf

    print('Training with different max lr ...')

    model = DenseNet(n_lbls if n_lbls != 2 else 1, pretrained=True,
                     freeze=True).to(device)  # .cuda()
    print('max_lr:', .01)
    best_loss = train_model(MAX_EPOCHS, model, best_loss=best_loss,
                            train_dl=train_dl, valid_dl=valid_dl,
                            max_lr=.01, wd=0, project_name=proj_id,
                            n_lbls=n_lbls,
                            save_path=save_path,
                            unfreeze_during_loop=(.1, .2))
    print('sending email to', email)
    send_email(receiver_name=name, receiver_email=email)


def predict_ml(project_id, paths, aspect_r, n_training_labels):
    predictions = []

    print('constructing model ...')
    model = DenseNet(out_size=n_training_labels
                     if n_training_labels != 2 else 1).to(device)
    print('loading model ...')
    load_model(model, CLIENT, project_id, bucket_name=BUCKET_ORIG,
               tmp_p=project_id+TMP_MODEL_FILE)
    print('predicting ...')
    model.eval()

    for idx, path in enumerate(paths):
        x = get_image(CLIENT, path, show=False,
                      bucket_name=BUCKET_ORIG)
        x = cv2.resize(x, (250, int(aspect_r*250)))

        x = x.astype(np.float32)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB) / 255

        center_crop(x, r_pix=R_PIX)

        x = normalize_imagenet(x)

        x = np.rollaxis(x, 2)

        x = torch.tensor(x).float().unsqueeze(0)

        y = model(x)

        logits = y.detach().cpu().numpy()[0]

        predictions.append(np.argmax(logits))

    return predictions


if __name__ == '__main__':
    pass
