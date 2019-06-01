Requirements
=============
The deploy script assumes that your remote  machine is accessible through the port speciffied in ``product-analytics-group-project-deepvision/code/.flaskenv``. It also assumes that the following packages have been previously installed.

* conda 4.6.8
* git version 2.19.1

And the ``git`` credentials of the user that is specified via command line at deployment time are stored in the remote machine.

The following packages are required (stated in `environment.yml`) as dependencies and will automatically be downloaded when running the deploy script:

 - alabaster=0.7.12
 - asn1crypto=0.24.0
 - babel=2.6.0
 - bcrypt=3.1.6
 - boto3=1.9.111
 - botocore=1.12.112
 - ca-certificates=2019.1.23
 - certifi=2019.3.9
 - cffi=1.12.2
 - chardet=3.0.4
 - click=7.0
 - cryptography=2.6.1
 - docutils=0.14
 - flask=1.0.2
 - flask-login=0.4.1
 - flask-sqlalchemy=2.3.2
 - flask-wtf=0.14.2
 - idna=2.8
 - imagesize=1.1.0
 - itsdangerous=1.1.0
 - jinja2=2.10
 - jmespath=0.9.4
 - libedit=3.1.20181209
 - libffi=3.2.1
 - libsodium=1.0.16
 - markupsafe=1.1.1
 - ncurses=6.1
 - openssl=1.1.1b
 - packaging=19.0
 - paramiko=2.4.2
 - pip=19.0.3
 - pyasn1=0.4.5
 - pycodestyle=2.5.0
 - pycparser=2.19
 - pygments=2.3.1
 - pynacl=1.3.0
 - pyopenssl=19.0.0
 - pyparsing=2.3.1
 - pysocks=1.6.8
 - python=3.7.1
 - python-dateutil=2.8.0
 - python-dotenv=0.10.1
 - pytz=2018.9
 - readline=7.0
 - requests=2.21.0
 - s3transfer=0.2.0
 - selenium=3.141.0
 - setuptools=40.8.0
 - simplejson=3.16.0
 - six=1.12.0
 - snowballstemmer=1.2.1
 - sphinx=1.8.5
 - sphinxcontrib=1.0
 - sphinxcontrib-websupport=1.1.0
 - sqlalchemy=1.3.1
 - sqlite=3.27.2
 - tk=8.6.8
 - urllib3=1.24.1
 - werkzeug=0.14.1
 - wheel=0.33.1
 - wtforms=2.2.1
 - xlrd=1.2.0
 - xz=5.2.4
 - zlib=1.2.11

And the following are packages installed through ``pip``:

   - appnope==0.1.0
   - attrs==19.1.0
   - backcall==0.1.0
   - bleach==3.1.0
   - decorator==4.4.0
   - defusedxml==0.5.0
   - entrypoints==0.3
   - gunicorn==19.9.0
   - ipykernel==5.1.0
   - ipython==7.4.0
   - ipython-genutils==0.2.0
   - ipywidgets==7.4.2
   - jedi==0.13.3
   - jsonschema==3.0.1
   - jupyter==1.0.0
   - jupyter-client==5.2.4
   - jupyter-console==6.0.0
   - jupyter-core==4.4.0
   - mistune==0.8.4
   - nbconvert==5.4.1
   - nbformat==4.4.0
   - notebook==5.7.6
   - pandocfilters==1.4.2
   - parso==0.3.4
   - pexpect==4.6.0
   - pickleshare==0.7.5
   - prometheus-client==0.6.0
   - prompt-toolkit==2.0.9
   - ptyprocess==0.6.0
   - pyrsistent==0.14.11
   - pyzmq==18.0.1
   - qtconsole==4.4.3
   - send2trash==1.5.0
   - terminado==0.8.1
   - testpath==0.4.2
   - tornado==6.0.1
   - traitlets==4.3.2
   - wcwidth==0.1.7
   - webencodings==0.5.1
   - widgetsnbextension==3.4.2

