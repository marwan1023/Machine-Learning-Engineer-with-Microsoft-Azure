# Machine Learning Engineer with Microsoft-Azure
### by Marwan saeed Alsharabbi

## Azure Workspace
![Azure Workspace](https://github.com/marwan1023/Machine-Learning-Engineer-with-Microsoft-Azure/blob/master/Project%231%20Optimizing-an-ML-Pipelin/Screenshots/run_an_experiment_as_a_pipeline.png)

## Optimizing an ML Pipeline in Azure

## Project Overview

In this project, you'll have the opportunity to create and optimize an ML pipeline. You'll be provided a custom-coded model—a standard Scikit-learn Logistic Regression—the hyperparameters of which you will optimize using HyperDrive. You'll also use AutoML to build and optimize a model on the same dataset, so that you can compare the results of the two methods.

You can see the main steps that you'll be taking in the diagram below:
![plan digram](https://github.com/marwan1023/Machine-Learning-Engineer-with-Microsoft-Azure/blob/master/Project%231%20Optimizing-an-ML-Pipelin/Screenshots/optimizing-an-ml-pipeline.png)


## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**

- we use the UCI Bank Marketing dataset to showcase how you can use AutoML for a classification problem and deploy it to an Azure Container Instance (ACI). The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.
- The dataset contains 32950 training examples in a csv file. https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

We had two approaches to solving the problem, the first approach was by using Hyperdrive to obtain the best values of the hyperparamters for a scikit-learn logistic regression model, in order to maximize the accuracy of the model.  
The second approach, was to use Azure's AutoML to find the best performing model based on the highest accuracy value. Accuracy : 0.9176403641881639
 * Azure's AutoML and using Hyperdrive
![automl](https://github.com/marwan1023/Machine-Learning-Engineer-with-Microsoft-Azure/blob/master/Project%231%20Optimizing-an-ML-Pipelin/Screenshots/Capture23.PNG)

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

* We first need to prepare our train.py script by:  
  * Importing the csv file containing the marketing campaigns data into our dataset using the TabularDatasetFactory module.  
  * Cleaning the dataset, which included droping NaN values.  
  * Splitting our dataset into training set (80% of the data) & test set (20% of the data.)   
  * Creating a Logistic Regression model using sci-kit learn.  
  * Creating a directory(outputs) to save the generated model into it.  
* After the train.py script is ready, we choose a proper parameter sampling method for the inverse regularization paramter(C) & the maximum number of iterations(max_iter), early termination policy and an estimator to create the HyperDriveConfig.  
  * The HyperDriveConfig was configured using the following:  
                             the estimator we created for the train.py,  
                             Paramater sampling method chosen,  
                             The early termination policy chosen,  
                             primary_metric_name, which is the Accuracy,  
                             primary_metric_goal, which is to maximize the primary metric,  
                             max_total_runs=12,  
                             max_concurrent_runs=4  
* Then we submit the hyperdrive run.  
* Once the run is complete, we choose the best run (the run that achieved the maximum accuracy) and save the model generated.  
 The best value of the Accuracy was found to be: **0.9113808801213961** 
First, we will start by optimizing a logistic regression model using HyperDrive. We start by setting up a training script train.py where we create a dataset, train and evaluate a logistic regression model from Scikit-learn. Then, we move on to the notebook and use HyperDrive to find optimal hyperparameters. HyperDrive Will help us to find the best hyperparameters of our model. Our final model from the Scikit-learn Pipeline is the output of the HyperDrive tunning.


**What are the benefits of the parameter sampler you chose?**
 #### Random sampling
We will use RandomParameterSampling which Defines random sampling over a hyperparameter search space to sample from a set of discrete values for max_iter hyperparameter and from a uniform distribution for C hyperparameter. This will make our hyperparameter tunning more efficient.
- Random sampling is used to randomly select a value for each hyperparameter, which can be a mix of discrete and continuous values
- Azure Machine Learning lets you automate hyperparameter tuning and run experiments in parallel to efficiently optimize hyperparameters.
- supports discrete and continuous hyperparameters.
- It supports early termination of low-performance runs.
- Some users do an initial search with random sampling and then refine the search space to improve results.
- usually ahieves better performance and it helps in discovering new hyperparameter values.
- In this sampling algorithm, parameter values are chosen from a set of discrete values or a distribution over a continuous range. Examples of functions you can use include: choice(*options), uniform(min_value, max_value), loguniform(min_value, max_value), normal(mu, sigma), and lognormal(mu, sigma).
 #### Grid sampling
- Grid sampling can only be employed when all hyperparameters are discrete, and is used to try every possible combination of parameters in the search space.
- supports discrete hyperparameters. Use grid sampling if you can budget to exhaustively search over the search space.
- Supports early termination of low-performance runs.
- Performs a simple grid search over all possible values. Grid sampling can only be used with choice hyperparameters.
#### Bayesian sampling 
- Bayesian sampling chooses hyperparameter values based on the Bayesian optimization algorithm, which tries to select parameter combinations that will result in improved performance from the previous selection.
- Bayesian sampling is recommended if you have enough budget to explore the hyperparameter space. For best results, we recommend a maximum number of runs greater than or equal to 20 times the number of hyperparameters being tuned.
- Bayesian sampling only supports choice, uniform, and quniform distributions over the search space
- You can only use Bayesian sampling with choice, uniform, and quniform parameter expressions, and you can't combine it with an early-termination policy.

**What are the benefits of the early stopping policy you chose?**

We will also use BanditPolicy which defines an early termination policy based on slack criteria, and a frequency and delay interval for evaluation. The slack_factor is the amount of slack allowed with respect to the best performing training run. The evaluation_interval is the frequency for applying the policy. Each time the training script logs the primary metric counts as one interval.
**Algorithm**   

Logistic Regression is a supervisied binary classification algorithm that predicts the probability of a target varaible, returning either 1 or 0 (yes or no).  
![Algorithm](https://github.com/marwan1023/Machine-Learning-Engineer-with-Microsoft-Azure/blob/master/Project%231%20Optimizing-an-ML-Pipelin/Screenshots/1.png)

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

We will optimize the model using AutoML. AutoMl explores a wide range of models like XGBoost, LightGBM, StackEnsemble and a lot more. So, it's more probable to find a better than logistic regression as we are exploring more models. The wining model for AutoML is VotingEnsemble model which involves summing the predictions made by multiple other classification models.

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

The AutoMl model is slightly better than the logistic regression model in accuracy with less than 0.0062594840667678. The wining model was VotingEnsemble model which involves summing the predictions made by multiple other classification models. While the architecture of the logistic regression has the advantage of being much simpler than the VotingEnsemble model. Logistic regression is weighted sum of input passed through the sigmoid activation function,and the AutoML didn't require a train.py script.
 * Azure's AutoML 
 ![Azure's AutoML](https://github.com/marwan1023/Machine-Learning-Engineer-with-Microsoft-Azure/blob/master/Project%231%20Optimizing-an-ML-Pipelin/Screenshots/Capture20.PNG)
  * we can see the explanations are directly impacting the model
  ![epl1](https://github.com/marwan1023/Machine-Learning-Engineer-with-Microsoft-Azure/blob/master/Project%231%20Optimizing-an-ML-Pipelin/Screenshots/Capture10.PNG)
   ![epl2](https://github.com/marwan1023/Machine-Learning-Engineer-with-Microsoft-Azure/blob/master/Project%231%20Optimizing-an-ML-Pipelin/Screenshots/Capture12.PNG)
   * we can see the Metrics
   ![met1](https://github.com/marwan1023/Machine-Learning-Engineer-with-Microsoft-Azure/blob/master/Project%231%20Optimizing-an-ML-Pipelin/Screenshots/Capture19.PNG)
   ![met2](https://github.com/marwan1023/Machine-Learning-Engineer-with-Microsoft-Azure/blob/master/Project%231%20Optimizing-an-ML-Pipelin/Screenshots/Capture14.PNG)
   ![met3](https://github.com/marwan1023/Machine-Learning-Engineer-with-Microsoft-Azure/blob/master/Project%231%20Optimizing-an-ML-Pipelin/Screenshots/Capture15.PNG)
   ![met4](https://github.com/marwan1023/Machine-Learning-Engineer-with-Microsoft-Azure/blob/master/Project%231%20Optimizing-an-ML-Pipelin/Screenshots/Capture16.PNG)
   ![met5](https://github.com/marwan1023/Machine-Learning-Engineer-with-Microsoft-Azure/blob/master/Project%231%20Optimizing-an-ML-Pipelin/Screenshots/Capture17.PNG)
  
 * using Hyperdrive
 
  ![Hyperdrive](https://github.com/marwan1023/Machine-Learning-Engineer-with-Microsoft-Azure/blob/master/Project%231%20Optimizing-an-ML-Pipelin/Screenshots/Capture3.PNG)
  ![Hyperdrive1](https://github.com/marwan1023/Machine-Learning-Engineer-with-Microsoft-Azure/blob/master/Project%231%20Optimizing-an-ML-Pipelin/Screenshots/Capture22.PNG)
  
  After that, I think AutoMl is better than HyperDrive because with HyperDrive we only test one algorithm and with AutoML we can test several algorithms and choose the best one. In this case the accuracy is pretty similar, but with AutoMl the result is better
## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

we've seen how to do a simple Scikit-learn training run using the SDK, let's see if we can further improve the accuracy of our model. We can optimize our model's hyperparameters using Azure Machine Learning's hyperparameter tuning capabilities.

One of the improvements will be preprocess the data better for both Scikit-learn and AutoML and check if results is better. Also, for Scikit-learn I can try the following:

- Other combination of values for the parameters --C and --Max_iter. 

  This will define a search space with two parameters, init_lr and hidden_size. The init_lr can have a uniform distribution with  as a minimum value , maximum value, and the hidden_size will be a choice of values. 
  
- Try different classification algorithms ,classification is a form of “pattern recognition,” with classification algorithms applied to the training data to find the same pattern  in future sets of data. Some algorithms are specifically designed for binary classification and do not natively support more than two classes Because our dataset is binary, we can try another classification algorithms to improve the accuracy of the model and be more efficient

- try use BayesianSampling method as the data we have is not that large so an early termination policy is not really necessary, If no policy is specified, the hyperparameter tuning service will let all training runs execute to completion.

- One of the future experiments I will conduct is to try different primary performance metrics. For example, using AUC_weighted, average_precision_score_weighted, or precision_score_weighted. Choosing the right evaluation metric for the problem will be helpful for model training. Judging a model bu only Accuracy is not always optimal
    also useful to evaluate a model. So using different metric will be beneficial.
  - Machine Learning Engineer with Microsoft Azure is optimized using code or no code
    
## Proof of cluster clean up

Before closing the experiment, I cleaned up the resources I used on Azure cloud.

![Image of cluster marked for deletion](https://github.com/marwan1023/Machine-Learning-Engineer-with-Microsoft-Azure/blob/master/Project%231%20Optimizing-an-ML-Pipelin/Screenshots/Capture21.PNG)


