# Machine-Learning-Engineer-with-Microsoft-Azure
### Workspace
A machine learning workspace is the top-level resource for Azure Machine Learning.
![architecture](https://github.com/marwan1023/Machine-Learning-Engineer-with-Microsoft-Azure/blob/master/Project%232%20Operationalizing%20Machine%20Learning/Screenshots/architecture.PNG)

A taxonomy of the workspace is illustrated in the following diagram:
![taxonomy](https://github.com/marwan1023/Machine-Learning-Engineer-with-Microsoft-Azure/blob/master/Project%232%20Operationalizing%20Machine%20Learning/Screenshots/azure-machine-learning-taxonomy.png)

## Project#2 Operationalizing Machine Learning

In this project, I will continue to work with the Bank Marketing dataset. You will use Azure to configure a cloud-based machine learning production model, deploy it, and consume it. You will also create, publish, and consume a pipeline.

## Architectural Diagram and main steps
In this project, you will following the below steps:

- Authentication
- Automated ML Experiment
- Deploy the best model
- Enable logging
- Swagger Documentation
- Consume model endpoints
- Create and publish a pipeline
- Documentation

# Architectural Diagram 

![Architectural Diagram](https://github.com/marwan1023/Machine-Learning-Engineer-with-Microsoft-Azure/blob/master/Project%232%20Operationalizing%20Machine%20Learning/Screenshots/screen-shot-2020.png)

Both the Azure ML Studio and the Python SDK will be used in this project. You will start with authentication and then run an Automated ML experiment to deploy the best model.

Next, you will enable Application Insight to review important information about the service when consuming the model.

And finally, you will create, publish, and interact with a pipeline.

 DevOps: A set of best practices that helps provide continuous delivery of software at the highest quality with a constant feedback loop.

## Step 1 - Authentication
Authentication is crucial for the continuous flow of operations. Continuous Integration and Delivery system (CI/CD) rely on uninterrupted flows. When authentication is not set properly, it requires human interaction and thus, the flow is interrupted. An ideal scenario is that the system doesn't stop waiting for a user to input a password. So whenever possible, it's good to use authentication with automation.
### Authentication types
### Key- based
- Azure Kubernetes service enabled by default
- Azure Container Instances service disabled by default
### Token- based
- Azure Kubernetes service disabled by default
- Not support Azure Container Instances
#### Interactive
Used by local deployment and experimentation (e.g. using Jupyter notebook)
#### Service Principal
A “Service Principal” is a user role with controlled permissions to access specific resources. Using a service principal is a great way to allow authentication while reducing the scope of permissions, which enhances security.
New terms
CI/CD: Continuous Integration and Continuous Delivery platform. Jenkins, CircleCI, and Github Actions are a few examples.

see the Link : https://docs.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication

## Step 2- Automated ML Experiment
At this point, security is enabled and authentication is completed. In this step, you will create an experiment using Automated ML, configure a compute cluster, and use that cluster to run the experiment.
I will use the same Bankmarketing dataset with course 1.

Copy the link to a new browser window to download the data:

https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv

Upload the bankmarketing_train.csv to Azure Machine Learning Studio so that it can be used when training the model.

### Part 1: Configure deployment settings
 - Create a new Automated ML run
 - Next, make sure you have the dataset uploaded
 - Create and configure your new compute cluster.
 - Once the new compute cluster is successfully created, use this cluster to run the autoML experiment
 - You will see the experiment in the experiment section and a new model is created.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
