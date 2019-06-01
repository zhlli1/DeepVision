
# ec2_address = "ec2-54-191-79-88.us-west-2.compute.amazonaws.com"
ec2_address = "ec2-3-212-208-200.compute-1.amazonaws.com"

# key_file = "./deepvision.pem"
key_file = "./deepVision.pem"
# key_file = "./deepVision_m.pem"

user = "ec2-user"

# without /
git_repo_name = "product-analytics-group-project-deepvision"
git_user_id = "MSDS698"  # repo creator


def git_credentials():
    git_user = input('Input your git user:\n')
    return git_user
