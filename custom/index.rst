.. ImageAI documentation master file, created by
   sphinx-quickstart on Tue Jun 12 06:13:09 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Custom Training: Prediction
======================================

.. figure:: ../image6.jpg

**ImageAI** provides very powerful yet easy to use classes to train state-of-the-art deep learning algorithms like **SqueezeNet** , **ResNet**
, **InceptionV3** and **DenseNet** on your own image datasets using as few as **5 lines of code** to generate your own custom models .
Once you have trained your own custom model, you can use the **CustomImagePrediction** class provided by **ImageAI** to use your own models to recognize/predict 
any image or set of images.



**======= imageai.Classification.Custom.ClassificationModelTrainer =======**



The **ClassificationModelTrainer** class allows you to train any of the 4 supported deep learning algorithms (**MobileNetV2** , **ResNet50**
, **InceptionV3** and **DenseNet121**) on your own image dataset to generate your own custom models. Your image dataset must contain at least
2 different classes/types of images (e.g cat and dog) and you must collect at least 500 images for each of the classes to achieve maximum accuracy.

The training process generates a JSON file that maps the objects types in your image dataset and creates lots of models. 
You will then peak the model with the highest accuracy and perform custom image prediction using the model and the JSON file generated. 

Because model training is a compute intensive tasks, we strongly advise you perform this experiment using a computer with a NVIDIA GPU and the GPU version of Tensorflow installed. Performing model training on CPU will my take hours or days. With NVIDIA GPU powered computer system, this will take a few hours. You can use Google Colab for this experiment as it has an NVIDIA K80 GPU available.
To train a custom prediction model, you need to prepare the images you want to use to train the model. You will prepare the images as follows: 

 -- Create a dataset folder with the name you will like your dataset to be called (e.g pets) 
 
 -- In the dataset folder, create a folder by the name train 
 
 -- In the dataset folder, create a folder by the name test 
 
 -- In the train folder, create a folder for each object you want to the model to predict and give the folder a name that corresponds to the respective object name (e.g dog, cat, squirrel, snake) 
 
 -- In the test folder, create a folder for each object you want to the model to predict and give the folder a name that corresponds to the respective object name (e.g dog, cat, squirrel, snake) 
 
 -- In each folder present in the train folder, put the images of each object in its respective folder. This images are the ones to be used to train the model 
 
 -- To produce a model that can perform well in practical applications, I recommend you about 500 or more images per object. 1000 images per object is just great 
 
 -- In each folder present in the test folder, put about 100 to 200 images of each object in its respective folder. These images are the ones to be used to test the model as it trains 
 
 -- Once you have done this, the structure of your image dataset folder should look like below ::

    pets//train//dog//dog-train-images
    pets//train//cat//cat-train-images
    pets//train//squirrel//squirrel-train-images
    pets//train//snake//snake-train-images

    pets//test//dog//dog-test-images
    pets//test//cat//cat-test-images
    pets//test//squirrel//squirrel-test-images
    pets//test//snake//snake-test-images


Once your dataset is ready, you can proceed to creating an instance of the **ModelTraining** class. Find the example below ::

    from imageai.Classification.Custom import ClassificationModelTrainer

    model_trainer = ClassificationModelTrainer()


Once you have created an instance above, you can use the functions below to set your instance property and start the traning process.


* **.setModelTypeAsMobileNetV2()** , This function sets the model type of the training instance you created to the **MobileNetV2** model, which means the **MobileNetV2** algorithm will be trained on your dataset.  Find example code below :: 

    model_trainer.setModelTypeAsMobileNetV2()


* **.setModelTypeAsResNet50()** , This function sets the model type of the training instance you created to the **ResNet50** model, which means the **ResNet50** algorithm will be trained on your dataset.  Find example code below :: 

    model_trainer.setModelTypeAsResNet()


* **.setModelTypeAsInceptionV3()** , This function sets the model type of the training instance you created to the **InceptionV3** model, which means the **InceptionV3** algorithm will be trained on your dataset.  Find example code below :: 

    model_trainer.setModelTypeAsInceptionV3()


* **.setModelTypeAsDenseNet121()** , This function sets the model type of the training instance you created to the **DenseNet121** model, which means the **DenseNet121** algorithm will be trained on your dataset.  Find example code below :: 

    model_trainer.setModelTypeAsDenseNet121()

* **.setDataDirectory()** , This function accepts a string which must be the path to the folder that contains the **test** and **train** subfolder of your image dataset. Find example code,and parameters of the function below ::

    prediction.setDataDirectory(r"C:/Users/Moses/Documents/Moses/AI/Custom Datasets/pets")

 -- *parameter* **data_directory** (required) : This is the path to the folder that contains your image dataset.


* **.trainModel()** , This is the function that starts the training process. Once it starts, it will create a JSON file in the **dataset/json** folder which contains the mapping of the classes of the dataset. The JSON file  will be used during custom prediction to produce reults. Find exmaple code below ::

    model_trainer.trainModel(num_experiments=100, batch_size=32)


 -- *parameter* **num_experiments** (required) : This is the number of times the algorithm will be trained on your image dataset. The accuracy of your training does increases as the number of times it trains increases. However, it does peak after a certain number of trainings;and that point depends on the size and nature of the dataset.
 
 -- *parameter* **batch_size** (optional) : During training, the algorithm is trained on a set of images in parallel. Because of this, the default value is set to 32. You can increase or reduce this value if you understand well enough to know the capacity of the system you are using to train. Should you intend to chamge this value, you should set it to values that are in multiples of 8 to optimize the training process.


**Sample Code for Custom Model Training**


Find below a sample code for training custom models for your image dataset ::

    from imageai.Classification.Custom import ClassificationModelTrainer

    model_trainer = ClassificationModelTrainer()
    model_trainer.setModelTypeAsResNet50()
    model_trainer.setDataDirectory(r"C:/Users/Moses/Documents/Moses/AI/Custom Datasets/pets")
    model_trainer.trainModel(num_objects=10, num_experiments=100, enhance_data=True, batch_size=32, show_network_summary=True)


Below is a sample of the result when the training starts ::

    
    ==================================================
    Training with GPU
    ==================================================
    Epoch 1/100
    ----------
    100%|█████████████████████████████████████████████████████████████████████████████████| 282/282 [02:15<00:00,  2.08it/s]
    train Loss: 3.8062 Accuracy: 0.1178
    100%|███████████████████████████████████████████████████████████████████████████████████| 63/63 [00:26<00:00,  2.36it/s]
    test Loss: 2.2829 Accuracy: 0.1215
    Epoch 2/100
    ----------
    100%|█████████████████████████████████████████████████████████████████████████████████| 282/282 [01:57<00:00,  2.40it/s]
    train Loss: 2.2682 Accuracy: 0.1303
    100%|███████████████████████████████████████████████████████████████████████████████████| 63/63 [00:20<00:00,  3.07it/s]
    test Loss: 2.2388 Accuracy: 0.1470


Let us explain the details shown above: 

1. The line Epoch 1/100 means the network is training the first experiment of the targeted 100 

2. The line 1/25 [>.............................] - ETA: 52s - loss: 2.3026 - acc: 0.2500 represents the number of batches that has been trained in the present experiment

3. The best model is automatically saved to <dataset-directory>/models>

Once you are done training your custom model, you can use the **CustomImagePrediction** class described below to perform image prediction with your model. 



**======= imageai.Classification.Custom.CustomImageClassification =======**


This class can be considered a replica of the **imageai.Classification.CustomImageClassification** as it has all the same functions, parameters and results. The 
only differences are that this class works with your own trained model, you will need to specify the path to the JSON file generated during the training
and will need to specify the number of classes in your image dataset when loading the model. Below is an example of creating an instance of the class ::

    from imageai.Classification.Custom import CustomImageClassification
    
    prediction = CustomImageClassification()

Once you have created the new instance, you can use the functions below to set your instance property and start recognizing 
objects in images.

* **.setModelTypeAsMobileNetV2()** , This function sets the model type of the image recognition instance you created to the **MobileNetV2** model, which means you will be performing your image prediction tasks using the "MobileNetV2" model generated during your custom training.  Find example code below :: 

    prediction.setModelTypeAsMobileNetV2()


* **.setModelTypeAsResNet50()** , This function sets the model type of the image recognition instance you created to the **ResNet50** model, which means you will be performing your image prediction tasks using the "ResNet" model model generated during your custom training. Find example code below ::

    prediction.setModelTypeAsResNet50()


* **.setModelTypeAsInceptionV3()** , This function sets the model type of the image recognition instance you created to the **InecptionV3** model, which means you will be performing your image prediction tasks using the "InceptionV3" model generated during your custom training.  Find example code below ::

    prediction.setModelTypeAsInceptionV3()


* **.setModelTypeAsDenseNet121()** , This function sets the model type of the image recognition instance you created to the **DenseNet121** model, which means you will be performing your image prediction tasks using the "DenseNet" model generated during your custom training. Find example code below ::

    prediction.setModelTypeAsDenseNet121()


* **.setModelPath()** , This function accepts a string which must be the path to the model file generated during your custom training and must corresponds to the model type you set for your image prediction instance. Find example code,and parameters of the function below ::

    prediction.setModelPath("resnet50-idenprof-test_acc_0.78200_epoch-91.pt")

 -- *parameter* **model_path** (required) : This is the path to your downloaded model file.

* **.setJsonPath()** , This function accepts a string which must be the path to the JSON file generated during your custom training. Find example code and parameters of the function below ::

    prediction.setJsonPath("idenprof_model_classes.jsonn")

 -- *parameter* **model_path** (required) : This is the path to your downloaded model file.


* **.loadModel()** , This function loads the model from the path you specified in the function call above into your image prediction instance. You will have to set the parameter **num_objects** to the number of classes in your image dataset. Find example code and parameter details below ::

    prediction.loadModel()

* **.classifyImage()** , This is the function that performs actual prediction of an image. It can be called many times on many images once the model as been loaded into your prediction instance. Find example code,parameters of the function and returned values below ::

    predictions, probabilities = prediction.classifyImage("image1.jpg", result_count=2)

 -- *parameter* **image_input** (required) : This refers to the path to your image file, Numpy array of your image or image file stream of your image, depending on the input type you specified.

 -- *parameter* **result_count** (optional) : This refers to the number of possible predictions that should be returned. The parameter is set to 5 by default.

 -- *returns* **prediction_results** (a python list) : The first value returned by the **predictImage** function is a list that contains all the possible prediction results. The results are arranged in descending order of the percentage probability.

 -- *returns* **prediction_probabilities** (a python list) : The second value returned by the **predictImage** function is a list that contains the corresponding percentage probability of all the possible predictions in the **prediction_results**. 

* **.useGPU()** , This function loads the model in CPU and forces processes to be done on the CPU. This is because by default, ImageAI will use GPU/CUDA if available else default to CPU. Find example code::

    prediction.useGPU()


**Sample Codes**

Find below sample code for custom prediction ::

    from imageai.Classification.Custom import CustomImageClassification
    import os

    execution_path = os.getcwd()

    prediction = CustomImageClassification()
    prediction.setModelTypeAsResNet50()
    prediction.setModelPath(os.path.join(execution_path, "resnet50-idenprof-test_acc_0.78200_epoch-91.pt"))
    prediction.setJsonPath(os.path.join(execution_path, "idenprof_model_classes.json"))
    prediction.loadModel()

    predictions, probabilities = prediction.classifyImage(os.path.join(execution_path, "4.jpg"), result_count=5)

    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction , " : " , eachProbability)




.. toctree::
   :maxdepth: 2
   :caption: Contents:






   


