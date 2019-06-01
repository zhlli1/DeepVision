Tutorial
=========

The application is to be launched on an EC2 instance. You can either use your own EC2 instance or the EC2 instance of DeepVision team.

To run the application on your own instance:


    1. Edit the ``product-analytics-group-project-deepvision/code/user_definition.py`` file: Replace the current ``ec2_address`` with your instance's ec2 public DNS address. Replace the current ``key_file`` path with the path  to your own pem file to access your instance.

    2. Afterward, in the command line, go to ``product-analytics-group-project-deepvision/code/`` and run ``$ python deploy_script.py``. 

    3. Input your GitHub username, which should be the same one stored on the remote machine. The webpage should be running on the specified ec2 address and the port specified by ``FLASK_RUN_PORT`` in ``.flaskenv``.

If you would like to run the instance on our EC2:

	1. Keep the ``user_definition.py`` file as we provide. 

	2. Afterward, in the command line, go to ``product-analytics-group-project-deepvision/code/`` and run ``$ python deploy_script.py``.

	3. When prompted for Github username, type 'haivule' as the Github user.

	4. Then access the web application at http://3.19.63.10:8080/

We assume that the conda environment points to a ``.conda`` file that is in the home directory of the user environment.





