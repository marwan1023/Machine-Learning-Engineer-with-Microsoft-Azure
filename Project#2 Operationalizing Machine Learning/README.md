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

more information  in the Reference :[Authentication](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication)

## Step 2- Automated ML Experiment
At this point, security is enabled and authentication is completed. In this step, you will create an experiment using Automated ML, configure a compute cluster, and use that cluster to run the experiment.
I will use the same Bankmarketing dataset with course 1.
### How AutoML works
During training, Azure Machine Learning creates a number of pipelines in parallel that try different algorithms and parameters for you. The service iterates through ML algorithms paired with feature selections, where each iteration produces a model with a training score. The higher the score, the better the model is considered to "fit" your data. It will stop once it hits the exit criteria defined in the experiment.
 more information  in the Reference :[automated-ml](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-first-experiment-automated-ml)

###  Configure deployment settings
 - Create a new Automated ML run
 - Next, make sure you have the dataset uploaded  Copy the link to a new browser window to download the data:  https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv
   Upload the bankmarketing_train.csv to Azure Machine Learning Studio so that it can be used when training the model.
 - Create and configure your new compute cluster.
 For experiment workloads with high scalability requirements, you can use Azure Machine Learning compute clusters; which are multi-node clusters of Virtual Machines that  automatically scale up or down to meet demand. This is a cost-effective way to run experiments that need to handle large volumes of data or use parallel processing to distribute the workload and reduce the time it takes to run.
 
 - Once the new compute cluster is successfully created, use this cluster to run the autoML experiment
       more information  in the Reference :[compute-target](https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-target)
    
 - You will see the experiment in the experiment section and a new model is created.
 
 ## Step 3: Deploy the Best Model
 After the experiment run completes, a summary of all the models and their metrics are shown, including explanations. The Best Model will be shown in the Details tab. In the   Models tab, it will come up first (at the top). Make sure you select the best model for deployment.

Deploying the Best Model will allow to interact with the HTTP API service and interact with the model by sending data over POST requests.

- Go to the Automated ML section and find the recent experiment with a completed status. Click on it.
- Go to the "Model" tab and select a model from the list and click it. Above it, a triangle button (or Play button) will show with the "Deploy" word. Click on it.
   1) Fill out the form with a meaningful name and description. For Compute Type use Azure Container Instance (ACI)

   2) Enable Authentication

   3) Do not change anything in the Advanced section.
- Deployment takes a few seconds. After a successful deployment, a green checkmark will appear on the "Run" tab and the "Deploy status" will show as succeed.

## Step 4: Enable Application Insights
Application Insights is an Azure service that helps you to monitor the performance and behavior of web applications.
It mostly captures two kinds of data: events and metrics. Events are individual data points that can represent any kind of event that occurs in an app. These events can be technical events that occur within the application runtime or those that are related to the business domain of the application or actions taken by users. Metrics are measurements of values, typically taken at regular intervals, that aren't tied to specific events. Like events, metrics can be related to the application's runtime or infrastructure (like the length of a queue) or related to the application's business domain or users (like how many videos are viewed in an hour).

 more information  in the Reference
 [Enable-application-insights](https://docs.microsoft.com/en-us/learn/modules/capture-page-load-times-application-insights/2-enable-application-insights)
 [Enable-application-insights](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-enable-app-insights)
 
 ## Step: Swagger Documentation
 In this step, you will consume the deployed model using Swagger.

  Azure provides a Swagger JSON file for deployed models. Head to the Endpoints section, and find your deployed model there, it should be the first one on the list.

  A few things you need to pay attention to:

- Ensure that Docker is installed on your computer.
- Azure provides a Swagger JSON file for deployed models. Head to the Endpoints section, and find your deployed model there.
- Click on the name of the model, and details will open that contains a Swagger URI section. Download the file locally to your computer and put it in the same directory with serve.py and swagger.sh.

- Run two scripts:

  1-serve.py will start a Python server on port 8000. This script needs to be right next to the downloaded swagger.json file. NOTE: this will not work if swagger.json is not on the same directory.

  2-swagger.sh which will download the latest Swagger container, and it will run it on port 80. If you don't have permissions for port 80 on your computer, update the script to a higher number (above 9000 is a good idea).

- Open the browser and go to http://localhost:8000 where serve.py should list the contents of the directory. swagger.json must show. If it doesn't, it needs to be downloaded from the deployed model endpoint.
- Go to http://localhost/ which should have Swagger running from the container (as defined in swagger.sh). If you changed the port number, use that new port number to reach the local Swagger service (for example, http://localhost:9000 if port 9000 is used).
- On the top bar, where petsore.swagger.io shows, change it to http://localhost:8000/swagger.json, then hit the Explore button. It should now display the contents of the API for the model
 - Look around at the different HTTP requests that are supported for the model
 
## Step 6: Consume Model Endpoints

Once the model is deployed, use the endpoint.py script provided to interact with the trained model. In this step, you need to run the script, modifying both the scoring_uri and the key to match the key for your service and the URI that was generated after deployment. in the Reference ["How to consume a web service"](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-consume-web-service?tabs=python)

- In Azure ML Studio, head over to the "Endpoints" section and find a previously deployed model. The compute type should be ACI (Azure Container Instance).

- In the "Consume" tab, of the endpoint, a "Basic consumption info" will show the endpoint URL and the authentication types. Take note of the URL and the "Primary Key" authentication type.
- Using the provided endpoint.py replace the scoring_uri and key to match the REST endpoint and primary key respectively. The script issues a POST request to the deployed model and gets a JSON response that gets printed to the terminal.

### Benchmark the Endpoint
- Make sure you have the Apache Benchmark command-line tool installed and available in your path:
- Run the endpoint.py. Just like before, it is important to use the right URI and Key to communicate with the deployed endpoint. A data.json should be present. This is required for the next step where the JSON file is used to HTTP POST to the endpoint.
- In the provided started code, there is a benchmark.sh script with a call to ab similar to this:

 ab -n 10 -v 4 -p data.json -T 'application/json' -H 'Authorization: Bearer SECRET' http://URL.azurecontainer.io/score 

## Step 7: Create, Publish and Consume a Pipeline

 Automation is a core pillar of DevOps applicable to Machine Learning operations.

A good feature of Azure is Pipelines, and these are closely related to automation. Some key factors covered about pipelines are:
#### Creating a pipeline 

When creating a Pipeline. Pipelines can take configuration and different steps and there are areas you can play with when creating a pipeline 
 - Batch inference: The process of doing predictions using parallelism. In a pipeline, it will usually be on a recurring schedule
 - Recurring schedule: A way to schedule pipelines to run at a given interval
 - Pipeline parameters: Like variables in a Python script, these can be passed into a script argument
  
#### Publishing a pipeline

Publishing a pipeline is the process of making a pipeline publicly available. You can publish pipelines in Azure Machine Learning Studio, but you can also do this with the Python SDK.

When a Pipeline is published, a public HTTP endpoint becomes available, allowing other services, including external ones, to interact with an Azure Pipeline.

##### Automation with pipelines
Pipelines are all about Automation. Automation connects different services and actions together to be part of a new workflow that wasn’t possible before.

There are some good examples of how different services can communicate to the pipeline endpoint to enable automation.

- A hosted project using version control: when a new change gets merged, a trigger is created to send an HTTP request to the endpoint and train the model.
- A newer dataset gets uploaded to a storage system that triggers an HTTP request to the endpoint to re-train the model.
- Several teams that want to use AutoML with datasets that are hosted externally can configure the external cloud provider to trigger an HTTP request when a new dataset gets saved.
- A CI /CD platform like Jenkins, with a job that submits an HTTP request to Azure when it completes without error.
#### Consume Pipeline Endpoint (API)
Pipeline endpoints can be consumed via HTTP, but it is also possible to do so via the Python SDK. Since there are different ways to interact with published Pipelines, this makes the whole pipeline environment very flexible.

It is key to find and use the correct HTTP endpoint to interact with a published pipeline. Sending a request over HTTP to a pipeline endpoint will require authentication in the request headers.

Interacting with a pipeline via an HTTP API endpoint more information in the Reference ["What are Machine Learning Pipelines"](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-your-first-pipeline)

Use a Parallel Run Step in a pipeline. Reference: ["How to use parallel run stepin a pipeline"](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-parallel-run-step)

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
