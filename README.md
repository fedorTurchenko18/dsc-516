# Federated Learning Performance Evaluation for different models and libraries

## Description of Workflow Under `main.py`

The following diagram reflects the algorithm which is being executed once the user runs `main.py`. Chat GPT was used in order to generate the scheme.

<img src="https://snipboard.io/FhjbTr.jpg" alt="aws-credentials" width="300"/>

Below, one may find the extended walkthrough of the workflow.

### 1. Server Launch
The `main.py`, subsequently referred to as "the System", at first launches an ec2 instance that will serve as the Flower Server with a prespecified number of rounds. The instance environment set-up is declared at the strart-up script which is passed to the instance creation request from `./flower-dependencies/server/server_startup.txt`. This script exports various environmental variables, installs Git, Pip, and the requirements from `./flower-dependencies/server/requirements.txt`. Then it starts the server with the `./flower-dependencies/server/run_server.py` script. This step invlolves the creation of S3 bucket as well where the results of the simulation will be stored. The System waits for the complete server launch, ensuring that all checks were passed. Federated Learning workflow could not be simulated without properly working server which is launched and waiting for requests.

The server is being launched under the customized `ServerStrategy` class from `./flower-dependencies/server/strategy.py`. Essentially, it is a wrapper that allows for convenient pass of user-defined parameters, namely `--strategy` and `--n_client_instances` to be conveniently passed to the server strategy, as well as the server-side model evaluation to be available. The latter requires the extraction of validation set from the images data which is stored in the repository.

### 2. Clients Launch
Further action, conducted by the System is devoted to starting the client instances. Depending on the user-defined `--n_client_instances` parameter, the System will initialize EC2 instances that will serve as the clients for the Flower server. The environment set-up is conducted in the similar fashion as for server, being passed from `./flower-dependencies/client/client_startup.txt`. The distinction, though, is encompassed in the absence of the need to create S3 bucket and the launch waiting routine. The latter is defined by waiting only for the instances initialization, without ensuring pass of all checks. Since the number of clients could be quite considerable, it is inefficient performance-wise to synchroniously wait for the full startup of each client instance.

The clients are launched under custom `UniversalClient` class from `./flower-dependencies/client/client.py`. Training data is disseminated to the clients in equal, or approximately if impossible, proportions; however, the balance of target variable classes is not required as in real-life Federated Learning scenarios it is commonly not a case. CNN model, being another parameter of `UniversalClient` is defined at `./flower-dependencies/model.py` and built using the `keras_core` module, encompassing the opportunity to run under various backends. The model is passed to the server strategy as well due to the need in server-side evaluation.

### 3. Results Extraction

Once, the clients passed the whole startup and check procedures, the Federated Learning simulation commences. Both server and clients write logs locally on EC2 instances, and once the training for a certain strategy is being completed, these are pushed to S3 bucket. Server logs include the data related to Federated Learning, such as ML-related loss and metric, as well as the round-wise execution time. Client logs allow one to assess the CPU load in percentages as well as model fitting- and evaulation-wise execution time. The completeness of Federated Learning simulation for all desired strategies allows the System to download the log files from S3. Finally, it terminates the EC2 instances and checks for local presence of all required files; if all of these are avialable locally, the S3 bucket is emptied and deleted. Otherwise, it is up to user to clear it manually.

## Instruction for Running the Simulation
1. Create `.env` file locally
2. Copy-paste contents of `.env_copy` to `.env`
3. Start the AWS Academy Learning Lab
4. Replace the placeholder-like strings with real values from your AWS Academy Learner Lab session by clicking :information_source: AWS Details. Reference:
<p float="left">
  <img src="https://sun9-42.userapi.com/impg/DYEB3AT48yllPzWMAgYAVaWtj_-t5gPIz9k3pg/yupOWkYahvA.jpg?size=712x658&quality=95&sign=32e3fb5d15f710eac79f57db569af1fd&type=album" alt="aws-credentials" width="300"/>
  <img src="https://sun9-42.userapi.com/kg6MBo9vkjdDLAo1b4nhkRJGSh22_XYSivrHjw/dDXouXf0Kp4.jpg?size=688x636&quality=95&sign=d9d40bb5e17a5e30a9f33657fc7d6899&type=album" alt="aws-region" width="300"/>
</p>

6. [Install AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

7. Set up the AWS Key Pair by running:

`bash export_aws_credentials.bash`

7. Install the requirements:

`pip install -r requirements.txt`

8. Run `python main.py --help` to see all the parameters available for simulation
   
   Required ones:
     - `--backend`
     - `--n_client_instances`
10. Run `python main.py` with desired parameters
