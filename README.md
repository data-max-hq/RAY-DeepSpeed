# Ray Project Setup

This project uses Ray to manage AWS EC2 instances for distributed computing. Following up on our previous work on Ray, we will be utilizing DeepSpeed for its ZERO 3 optimizer to fine-tune on a Ray cluster. Tested and working using Python 3.11.

## Getting Started

1. **Clone the repository:**
   
   ```bash
   git clone ray-distributed-compute.git
   cd ray-distributed-compute
   ```

2. **Install Ray:**
   
   ```bash
   pip install ray
   ```

3. **Install DeepSpeed:**
   
   ```bash
   pip install deepspeed
   ```

4. **Configure AWS CLI:**
   
   ```bash
   aws configure
   ```

5. **Find Your AWS Account Number:**
   
   ```bash
   aws sts get-caller-identity --query Account --output text
   ```
   
   Note down your AWS account number, as you will need it for the next steps.

6. **Create an IAM role with full S3 access:**
   
   ```bash
   aws iam create-role --role-name ray-s3-fullaccess --assume-role-policy-document file://trust-policy.json
   ```

7. **Create an S3 bucket:**
   
   ```bash
   aws s3api create-bucket --bucket ray-bucket-model-output --region eu-central-1 --create-bucket-configuration LocationConstraint=eu-central-1
   ```

8. **Attach the S3 full access policy to the role:**
   
   ```bash
   aws iam attach-role-policy --role-name ray-s3-fullaccess --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
   ```

9. **Create an instance profile:**
   
   ```bash
   aws iam create-instance-profile --instance-profile-name ray-s3-instance-profile
   ```

10. **Add the role to the instance profile:**
   
    ```bash
    aws iam add-role-to-instance-profile --instance-profile-name ray-s3-instance-profile --role-name ray-s3-fullaccess
    ```

11. **Create a policy to allow `iam:PassRole` for `ray-autoscaler-v1`:**
    Replace `<your-account-id>` with your AWS account number.

    ```bash
    aws iam create-policy \
        --policy-name PassRolePolicy \
        --policy-document '{
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": "iam:PassRole",
                    "Resource": "arn:aws:iam::<your-account-id>:role/ray-s3-fullaccess"
                }
            ]
        }'
    ```

12. **Attach the `PassRolePolicy` to the `ray-autoscaler-v1` role:**
    
    ```bash
    aws iam attach-role-policy \
        --role-name ray-autoscaler-v1 \
        --policy-arn arn:aws:iam::<your-account-id>:policy/PassRolePolicy
    ```

13. **Retrieve the ARN for `ray-s3-instance-profile`:**
    
    ```bash
    aws iam list-instance-profiles-for-role --role-name ray-s3-fullaccess --query 'InstanceProfiles[0].Arn' --output text
    ```
    
    Note down the retrieved ARN for use in the next steps.

14. **Update the YAML configuration:**

    Open your `raycluster.yaml` file and replace the placeholder with the ARN you retrieved:

    ```yaml
    ray.worker.default:
      resources:
        CPU: 1
        resources: 15
      node_config:
        ImageId: ami-07652eda1fbad7432
        InstanceType: p3.2xlarge
        IamInstanceProfile:
          Arn: arn:aws:iam::<your-account-id>:instance-profile/ray-s3-instance-profile
    ```

15. **Start the Ray cluster:**
    
    ```bash
    ray up raycluster.yaml
    ```

16. **Access the Ray dashboard:**
    
    ```bash
    ray dashboard raycluster.yaml
    ```

17. **Submit a Ray job:**

    Open a new terminal window and navigate to your project directory:

    ```bash
    cd <project-directory>
    ```

    Submit the Ray job:

    ```bash
    ray job submit --address http://localhost:8265 --working-dir . -- python3 main.py
    ```

18. **Check the S3 bucket:**

    When the job finishes running, head over to the specified S3 bucket (`ray-bucket-model-output`) where you should find the trained model.

## Overview

Ray is a distributed computing framework that allows you to easily scale your applications across multiple machines. In this setup, you'll use Ray to manage a head node and a larger VM instance with a powerful CPU and 1 GPU, leveraging their respective hardware capabilities to perform computational tasks efficiently. Additionally, using DeepSpeed with the ZERO 3 optimizer will enhance your fine-tuning process.

For detailed documentation on Ray, visit the [Ray documentation](https://docs.ray.io/). For more on DeepSpeed, check out the [DeepSpeed documentation](https://www.deepspeed.ai/docs/).
