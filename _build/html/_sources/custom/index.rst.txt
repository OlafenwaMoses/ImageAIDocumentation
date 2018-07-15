.. ImageAI documentation master file, created by
   sphinx-quickstart on Tue Jun 12 06:13:09 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Custom Training and Prediction Classes
======================================
**ImageAI** provides very powerful yet easy to use classes to train state-of-the-art deep learning algorithms like **SqueezeNet** , **ResNet**
, **InceptionV3** and **DenseNet** on your own image datasets using as few as **5 lines of code** to generate your own custom models .
Once you have trained your own custom model, you can use the **CustomImagePrediction** class provided by **ImageAI** to use your own models to recognize/predict 
any image or set of images.



**======= imageai.Prediction.Custom.ModelTraining =======**



The **ModelTraining** class allows you to train any of the 4 supported deep learning algorithms (**SqueezeNet** , **ResNet**
, **InceptionV3** and **DenseNet**) on your own image dataset to generate your own custom models. Your image dataset must contain at least
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

    from imageai.Prediction.Custom import ModelTraining

    model_trainer = ModelTraining()


Once you have created an instance above, you can use the functions below to set your instance property and start the traning process.


* **.setModelTypeAsSqueezeNet()** , This function sets the model type of the training instance you created to the **SqueezeNet** model, which means the **SqueezeNet** algorithm will be trained on your dataset.  Find example code below :: 

    model_trainer.setModelTypeAsSqueezeNet()


* **.setModelTypeAsResNet()** , This function sets the model type of the training instance you created to the **ResNet** model, which means the **ResNet** algorithm will be trained on your dataset.  Find example code below :: 

    model_trainer.setModelTypeAsResNet()


* **.setModelTypeAsInceptionV3()** , This function sets the model type of the training instance you created to the **InceptionV3** model, which means the **InceptionV3** algorithm will be trained on your dataset.  Find example code below :: 

    model_trainer.setModelTypeAsInceptionV3()


* **.setModelTypeAsDenseNet()** , This function sets the model type of the training instance you created to the **DenseNet** model, which means the **DenseNet** algorithm will be trained on your dataset.  Find example code below :: 

    model_trainer.setModelTypeAsDenseNet()

* **.setDataDirectory()** , This function accepts a string which must be the path to the folder that contains the **test** and **train** subfolder of your image dataset. Find example code,and parameters of the function below ::

    prediction.setDataDirectory(r"C:/Users/Moses/Documents/Moses/AI/Custom Datasets/pets")

 -- *parameter* **data_directory** (required) : This is the path to the folder that comtaims your image dataset.


* **.trainModel()** , This is the function that starts the training process. Once it starts, it will create a JSON file in the **dataset/json** folder (e.g **pets/json**) which contains the mapping of the classes of the dataset. The JSON file  will be used during custom prediction to produce reults. Find exmaple code below ::

    model_trainer.trainModel(num_objects=4, num_experiments=100, enhance_data=True, batch_size=32, show_network_summary=True)


 -- *parameter* **num_objects** (required) : This refers to the number of different classes in your image dataset.
 
 -- *parameter* **num_experiments** (required) : This is the number of times the algorithm will be trained on your image dataset. The accuracy of your training does increases as the number of times it trains increases. However, it does peak after a certain number of trainings;and that point depends on the size and nature of the dataset.
 
 -- *parameter* **enhance_data** (optional) : This parameter is used to tranform your image dataset in order to generate more sample for training. It is set to False by default. However, it is useful to set it to True if your image dataset contains less than 1000 images per class.

 -- *parameter* **batch_size** (optional) : During training, the algorithm is trained on a set of images in parallel. Because of this, the default value is set to 32. You can increase or reduce this value if you understand well enough to know the capacity of the system you are using to train. Should you intend to chamge this value, you should set it to values that are in multiples of 8 to optimize the training process.

 -- *parameter* **show_network_summary** (optional) : This paramter when set to True displays the structure of the algorithm you are training on your image dataset in the CLI before training starts. It is set to False by default.
 
 -- *parameter* **initial_learning_rate** (optional) : This parameter is a highly technical value. It determines and control the behaviour of your training which is critical to the accuracy that can be achieved. Change this parameter's value only if you understand its function fully.

 -- *parameter* **initial_learning_rate** (optional) : This parameter is a highly technical value. It determines and control the behaviour of your training which is critical to the accuracy that can be achieved. Change this parameter's value only if you understand its function fully.

 -- *training_image_size* **initial_learning_rate** (optional) : This is the size at which the images in your image dataset will be trained, irrespective of their original sizes. The default value is 224 and must not be set to less than 100. Increasing this value increases accuracy but increases training time, and vice-versa.




**Sample Code for Custom Model Training**


Find below a sample code for training custom models for your image dataset ::

    from imageai.Prediction.Custom import ModelTraining

    model_trainer = ModelTraining()
    model_trainer.setModelTypeAsResNet()
    model_trainer.setDataDirectory(r"C:/Users/Moses/Documents/Moses/AI/Custom Datasets/pets")
    model_trainer.trainModel(num_objects=10, num_experiments=100, enhance_data=True, batch_size=32, show_network_summary=True)


Below is a sample of the result when the training starts ::

    Epoch 1/100
    1/25 [>.............................] - ETA: 52s - loss: 2.3026 - acc: 0.2500
    2/25 [=>............................] - ETA: 41s - loss: 2.3027 - acc: 0.1250
    3/25 [==>...........................] - ETA: 37s - loss: 2.2961 - acc: 0.1667
    4/25 [===>..........................] - ETA: 36s - loss: 2.2980 - acc: 0.1250
    5/25 [=====>........................] - ETA: 33s - loss: 2.3178 - acc: 0.1000
    6/25 [======>.......................] - ETA: 31s - loss: 2.3214 - acc: 0.0833
    7/25 [=======>......................] - ETA: 30s - loss: 2.3202 - acc: 0.0714
    8/25 [========>.....................] - ETA: 29s - loss: 2.3207 - acc: 0.0625
    9/25 [=========>....................] - ETA: 27s - loss: 2.3191 - acc: 0.0556
    10/25 [===========>..................] - ETA: 25s - loss: 2.3167 - acc: 0.0750
    11/25 [============>.................] - ETA: 23s - loss: 2.3162 - acc: 0.0682
    12/25 [=============>................] - ETA: 21s - loss: 2.3143 - acc: 0.0833
    13/25 [==============>...............] - ETA: 20s - loss: 2.3135 - acc: 0.0769
    14/25 [===============>..............] - ETA: 18s - loss: 2.3132 - acc: 0.0714
    15/25 [=================>............] - ETA: 16s - loss: 2.3128 - acc: 0.0667
    16/25 [==================>...........] - ETA: 15s - loss: 2.3121 - acc: 0.0781
    17/25 [===================>..........] - ETA: 13s - loss: 2.3116 - acc: 0.0735
    18/25 [====================>.........] - ETA: 12s - loss: 2.3114 - acc: 0.0694
    19/25 [=====================>........] - ETA: 10s - loss: 2.3112 - acc: 0.0658
    20/25 [=======================>......] - ETA: 8s - loss: 2.3109 - acc: 0.0625
    21/25 [========================>.....] - ETA: 7s - loss: 2.3107 - acc: 0.0595
    22/25 [=========================>....] - ETA: 5s - loss: 2.3104 - acc: 0.0568
    23/25 [==========================>...] - ETA: 3s - loss: 2.3101 - acc: 0.0543
    24/25 [===========================>..] - ETA: 1s - loss: 2.3097 - acc: 0.0625Epoch 00000: saving model to C:\Users\Moses\Documents\Moses\W7\AI\Custom Datasets\IDENPROF\idenprof-small-test\idenprof\models\model_ex-000_acc-0.100000.h5

    25/25 [==============================] - 51s - loss: 2.3095 - acc: 0.0600 - val_loss: 2.3026 - val_acc: 0.1000


Let us explain the details shown above: 

1. The line Epoch 1/100 means the network is training the first experiment of the targeted 100 

2. The line 1/25 [>.............................] - ETA: 52s - loss: 2.3026 - acc: 0.2500 represents the number of batches that has been trained in the present experiment

3. The line Epoch 00000: saving model to C:\Users\User\PycharmProjects\ImageAITest\pets\models\modelex-000acc-0.100000.h5 refers to the model saved after the present experiment. The ex_000 represents the experiment at this stage while the acc0.100000 and valacc: 0.1000 represents the accuracy of the model on the test images after the present experiment (maximum value value of accuracy is 1.0). This result helps to know the best performed model you can use for custom image prediction. 

Once you are done training your custom model, you can use the **CustomImagePrediction** class described below to perform image prediction with your model. 




**======= imageai.Prediction.Custom.CustomImagePrediction =======**


This class can be considered a replica of the **imageai.Prediction.ImagePrediction** as it has all the same functions, parameters and results. The 
only differences are that this class works with your own trained model, you will need to specify the path to the JSON file generated during the training
and will need to specify the number of classes in your image dataset when loading the model. Below is an example of creating an instance of the class ::

    from imageai.Prediction.Custom import CustomImagePrediction
    
    prediction = CustomImagePrediction()

Once you have created the new instance, you can use the functions below to set your instance property and start recognizing 
objects in images.

* **.setModelTypeAsSqueezeNet()** , This function sets the model type of the image recognition instance you created to the **SqueezeNet** model, which means you will be performing your image prediction tasks using the "SqueezeNet" model generated during your custom training.  Find example code below :: 

    prediction.setModelTypeAsSqueezeNet()


* **.setModelTypeAsResNet()** , This function sets the model type of the image recognition instance you created to the **ResNet** model, which means you will be performing your image prediction tasks using the "ResNet" model model generated during your custom training. Find example code below ::

    prediction.setModelTypeAsResNet()


* **.setModelTypeAsInceptionV3()** , This function sets the model type of the image recognition instance you created to the **InecptionV3** model, which means you will be performing your image prediction tasks using the "InceptionV3" model generated during your custom training.  Find example code below ::

    prediction.setModelTypeAsInceptionV3()


* **.setModelTypeAsDenseNet()** , This function sets the model type of the image recognition instance you created to the **DenseNet** model, which means you will be performing your image prediction tasks using the "DenseNet" model generated during your custom training. Find example code below ::

    prediction.setModelTypeAsDenseNet()


* **.setModelPath()** , This function accepts a string which must be the path to the model file generated during your custom training and must corresponds to the model type you set for your image prediction instance. Find example code,and parameters of the function below ::

    prediction.setModelPath("resnet_model_ex-020_acc-0.651714.h5")

 -- *parameter* **model_path** (required) : This is the path to your downloaded model file.

* **.setJsonPath()** , This function accepts a string which must be the path to the JSON file generated during your custom training. Find example code and parameters of the function below ::

    prediction.setJsonPath("model_class.json")

 -- *parameter* **model_path** (required) : This is the path to your downloaded model file.


* **.loadModel()** , This function loads the model from the path you specified in the function call above into your image prediction instance. You will have to set the parameter **num_objects** to the number of classes in your image dataset. Find example code and parameter details below ::

    prediction.loadModel(num_objects=4)

 -- *parameter* **num_objects** (required) : This must be set to the number of classes in your image dataset.

-- *parameter* **prediction_speed** (optional) : This parameter allows you to reduce the time it takes to predict in an image by up to 80% which leads to slight reduction in accuracy. This parameter accepts string values. The available values are "normal", "fast", "faster" and "fastest". The default values is "normal"



* **.predictImage()** , This is the function that performs actual prediction of an image. It can be called many times on many images once the model as been loaded into your prediction instance. Find example code,parameters of the function and returned values below ::

    predictions, probabilities = prediction.predictImage("image1.jpg", result_count=2)

 -- *parameter* **image_input** (required) : This refers to the path to your image file, Numpy array of your image or image file stream of your image, depending on the input type you specified.

 -- *parameter* **result_count** (optional) : This refers to the number of possible predictions that should be returned. The parameter is set to 5 by default.

 -- *parameter* **input_type** (optional) : This refers to the type of input you are parse into the **image_input** parameter. It is "file" by default and it accepts "array" and "stream" as well.


 -- *returns* **prediction_results** (a python list) : The first value returned by the **predictImage** function is a list that contains all the possible prediction results. The results are arranged in descending order of the percentage probability.

 -- *returns* **prediction_probabilities** (a python list) : The second value returned by the **predictImage** function is a list that contains the corresponding percentage probability of all the possible predictions in the **prediction_results**. 


* **.predictMultipleImages()** , This function can be used to perform prediction on 2 or more images at once. Find example code, parameters of the function and returned values below ::

    results_array = multiple_prediction.predictMultipleImages(all_images_array, result_count_per_image=2)

    for each_result in results_array:
        predictions, percentage_probabilities = each_result["predictions"], each_result["percentage_probabilities"]
        for index in range(len(predictions)):
            print(predictions[index] , " : " , percentage_probabilities[index])
        print("-----------------------")

  -- *parameter* **sent_images_array** (required) : This refers to a list that contains the path to your image files, Numpy array of your images or image file stream of your images, depending on the input type you specified.

  -- *parameter* **result_count_per_image** (optional) : This refers to the number of possible predictions that should be returned for each of the images. The parameter is set to 2 by default.

  -- *parameter* **input_type** (optional) : This refers to the format in which your images are in the list you parsed into the **sent_images_array** parameter. It is "file" by default and it accepts "array" and "stream" as well.


  -- *returns* **output_array** (a python list) : The value returned by the **predictMultipleImages** function is a list that contains dictionaries. Each dictionary correspondes
  the images contained in the array you parsed into the **sent_images_array**. Each dictionary has "prediction_results" property which is a list of athe prediction result for the image
  in that index as well as the "prediction_probabilities" which is a list of the corresponding percentage probability for each result.


**Sample Codes**

Find below sample code for custom prediction ::

    from imageai.Prediction.Custom import CustomImagePrediction
    import os

    execution_path = os.getcwd()

    prediction = CustomImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath(os.path.join(execution_path, "resnet_model_ex-020_acc-0.651714.h5"))
    prediction.setJsonPath(os.path.join(execution_path, "model_class.json"))
    prediction.loadModel(num_objects=4)

    predictions, probabilities = prediction.predictImage(os.path.join(execution_path, "4.jpg"), result_count=5)

    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction , " : " , eachProbability)








.. toctree::
   :maxdepth: 2
   :caption: Contents:






   


