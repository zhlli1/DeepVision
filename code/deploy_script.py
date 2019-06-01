import paramiko
import os
from os.path import expanduser
from user_definition import *

ENVIRONMENT_PATH = '/opt/conda/envs/deepVision'


def ssh_client():
    '''
    Return ssh client object

    :return: paramiko.SShClient()
    '''
    return paramiko.SSHClient()


def ssh_connection(ssh, ec2_address, user, key_file):
    """
    Establish an ssh connection.

    :param ssh: paramiko.SSHClient class
    :param ec2_address: (str) ec2 instance address
    :param user: (str) ssh username
    :param key_file: (str) location of the AWS
                     key from the root directory
    :return: None
    """
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ec2_address, username=user, key_filename=key_file)
    return ssh


def git_clone_pull(ssh, git_user_id, git_repo_name):
    """
    Clone/Updates 'git_repo_name' repository.

    :param ssh: paramiko.SSHClient class
    :return: None
    """
    stdin, stdout, stderr = ssh.exec_command("git --version")

    git_user = git_credentials()

    stdin, stdout, stderr = ssh.exec_command("git config " +
                                             "--global " +
                                             "credential.helper store")

    stdin, stdout, stderr = ssh.exec_command('cd ' + git_repo_name)

    # Try cloning the repo
    if b"" == stderr.read():

        git_pull_command = "cd " + git_repo_name + " ; git stash; git pull"
        stdin, stdout, stderr = ssh.exec_command(git_pull_command)

        print(stdout.read())
        print(stderr.read())

    else:
        git_clone_command = "git clone https://" + git_user +\
                            "@github.com/" + \
                            git_user_id + "/" + git_repo_name + ".git"

        stdin, stdout, stderr = ssh.exec_command(git_clone_command)
        print(stdout.read())
        print(stderr.read())


def create_or_update_environment(ssh, git_repo_name):
    """
    Creates/update python environment with the repo's .yaml file.

    :param ssh: paramiko.SSHClient class
    :return: None
    """

    stdin, stdout, stderr = ssh.exec_command(f"cd {ENVIRONMENT_PATH}")

    # Try cloning the repo
    if b"" != stderr.read():
        stdin, stdout, stderr = ssh.exec_command("conda env create -f " +
                                                 "~/" + git_repo_name +
                                                 "/" + "environment.yml")

        print(stdout.read())
        print(stderr.read())
    else:
        print('Updating environment...')
        stdin, stdout, stderr = ssh.exec_command("conda env update " +
                                                 "-f ~/" + git_repo_name +
                                                 "/" + "environment.yml")

        print(stdout.read())
        print(stderr.read())


def get_port(ssh, server_path):
    '''

    :param ssh: paramiko.SSHClient class
    :param server_path: path to the application directory
    (where ``.flaskenv`` is located)
    :return: (str) port number
    '''
    stdin, stdout, stderr = ssh.exec_command("cat " + os.path.join(server_path,
                                             '.flaskenv'))

    info = stdout.read().decode("utf-8").split('=')[-1]
    return info.strip()


def print_port(ssh, server_path):
    '''
    Prints the port number in which the app
    runs according to the .flaskenv file.

    :param ssh: paramiko ssh client (connected)
    :param server_path: path to the application directory
    (where ``.flaskenv`` is located)
    :return: None
    '''

    port = get_port(ssh, server_path)

    print("App running in port number " + port)


def launch_application(ssh, server_path='~/' + git_repo_name + '/code'):
    '''
    Launch application server_path under
    the deepVision environment and print port.

    :param ssh: paramiko ssh.Client (already connected)
    :param server_path: path to directory where run_app.py is located.
    :return: None
    '''

    # kill any process running from the app if any
    command = "kill -9 `ps aux |grep gunicorn |grep app | awk '{ print $2 }'` "
    stdin, stdout, stderr = ssh.exec_command(command)

    port = get_port(ssh, server_path)

    # run the server with the last version
    command = f"{ENVIRONMENT_PATH}/bin/gunicorn -D -b :{port} -w 20 " + \
              "--timeout 10000 --chdir " + \
              "product-analytics-group-project-deepvision/code/" + \
              " app:application"

    stdin, stdout, stderr = ssh.exec_command(command)

    # print(stdout.read())
    # print(stderr.read())

    print_port(ssh, server_path)


def close(ssh):
    """
    Closes the SSH connection.

    :param ssh: paramiko.SSHClient class
    :return: None
    """
    ssh.close()


def main():
    """
    Main function.

    :return:  None
    """

    ssh = ssh_client()
    ssh_connection(ssh, ec2_address, user, key_file)
    git_clone_pull(ssh, git_user_id, git_repo_name)
    create_or_update_environment(ssh, git_repo_name)
    # set_crontab(ssh)
    launch_application(ssh)
    close(ssh)


if __name__ == '__main__':
    main()
