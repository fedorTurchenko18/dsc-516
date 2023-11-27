# Federated Learning Performance Evaluation for different models and libraries

## Project Goals
The focus of the study was aimed at two key pillars:
1. Simulation of Federated Learning (FL) workflow on AWS EC2
2. Comparison of:
     - Backends for running Deep Learning (DL) model, namely Jax, Tensorflow, and Pytorch, combined with...
     - ...different FL Strategies, namely Federated- Averaging, Averaging with Momentum, Adaptive Gradient Optimizer, Adaptive Moment Estimation,...
     - ...in such aspects as:
       - General performance
       - Convergence speed over FL rounds
       - CPU utilization, %
       - Speed of FL workflow execution

## Instruction for Running the Simulation
1. Create `.env` file locally
2. Copy-paste contents of `.env_copy` to `.env`
3. Start the AWS Academy Learning Lab
4. Replace the placeholder-like strings with real values from your AWS Academy Learner Lab session by clicking :information_source: AWS Details. Reference:
<img src="https://sun9-42.userapi.com/impg/DYEB3AT48yllPzWMAgYAVaWtj_-t5gPIz9k3pg/yupOWkYahvA.jpg?size=712x658&quality=95&sign=32e3fb5d15f710eac79f57db569af1fd&type=album" alt="aws-credentials" width="300"/>
<img src="https://sun9-42.userapi.com/kg6MBo9vkjdDLAo1b4nhkRJGSh22_XYSivrHjw/dDXouXf0Kp4.jpg?size=688x636&quality=95&sign=d9d40bb5e17a5e30a9f33657fc7d6899&type=album" alt="aws-region" width="300"/>

5. Set up the AWS Key Pair by running:

`bash export_aws_credentials.bash`

6. Install the requirements:

`pip install -r requirements.txt`

7. Run `python main.py --help` to see all the parameters available for simulation (`--backend` is a mandatory argument)
> [!NOTE]
> - Project team simulates with all the rest parameters being set to their default values
> - @fedorTurchenko18 simulates with `--backend jax`
> - @gaturchenko simulates with `--backend torch`
> - @PanikosChristou99 simulates with `--backend tensorflow`
