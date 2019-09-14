.. ImageAI documentation master file, created by
   sphinx-quickstart on Tue Jun 12 06:13:09 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Custom Object Detection: Training and Inference
=================

.. figure:: ../image7.jpg

**ImageAI** provides the simple and powerful approach to training **custom object detection** models using the **YOLOv3** architeture. This allows you to train your own model on any set of images that corresponds to any type of object of interest. 

You can use your trained detection models to detect objects in images, videos and perform video analysis.



**======= imageai.Detection.Custom.DetectionModelTrainer =======**


This is the Detection Model training class, which allows you to train object detection models on image datasets that are in Pascal VOC annotation format, using the YOLOv3.
The training process generates a JSON file that maps the objects names in your image dataset and the detection anchors, as well as creates lots of models.

To get started, you need prepare your dataset in the **Pascal VOC Format** and organize it as detailed below:

-- Decide the type of object(s) you want to detect and collect about 200 (minimum recommendation) or more picture of each of the object(s)

-- Once you have collected the images, you need to annotate the object(s) in the images. You can use a tool like  `LabelIMG <https://github.com/tzutalin/labelImg>`_ to generate the annotations for your images.

-- Once you have the annotations for all your images, create a folder for your dataset **(E.g headsets)** and in this parent folder, create child folders **train** and **validation**

-- In the **train** folder, create **images** and **annotations** sub-folders. Put about 70-80% of your dataset of each object's images in the **images** folder and put the corresponding annotations for these images in the **annotations** folder.

-- In the **validation** folder, create **images** and **annotations** sub-folders. Put the rest of your dataset images in the **images** folder and put the corresponding annotations for these images in the **annotations** folder.

-- Once you have done this, the structure of your image dataset folder should look like below::

    >> train    >> images       >> img_1.jpg  (shows Object_1)
                >> images       >> img_2.jpg  (shows Object_2)
                >> images       >> img_3.jpg  (shows Object_1, Object_3 and Object_n)
                >> annotations  >> img_1.xml  (describes Object_1)
                >> annotations  >> img_2.xml  (describes Object_2)
                >> annotations  >> img_3.xml  (describes Object_1, Object_3 and Object_n)

    >> validation   >> images       >> img_151.jpg (shows Object_1, Object_3 and Object_n)
                    >> images       >> img_152.jpg (shows Object_2)
                    >> images       >> img_153.jpg (shows Object_1)
                    >> annotations  >> img_151.xml (describes Object_1, Object_3 and Object_n)
                    >> annotations  >> img_152.xml (describes Object_2)
                    >> annotations  >> img_153.xml (describes Object_1)



- You can train your custom detection model completely from scratch or use transfer learning (recommended for better accuracy) from a pre-trained YOLOv3 model. Also, we have provided a sample annotated Hololens and Headsets (Hololens and Oculus) dataset for you to train with. Download the pre-trained YOLOv3 model and the sample datasets in the link below

`Sample dataset and pre-trained YOLOv3 <https://github.com/OlafenwaMoses/ImageAI/releases/tag/essential-v4>`_


- For the purpose of training your detection model, we advice that you have the **Tensorflow-GPU v1.13.1** installed to avoid errors::

    pip3 install tensorflow-gpu==1.13.1


Below is the code to train new detection models on your dataset::

    from imageai.Detection.Custom import DetectionModelTrainer

    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory="hololens")
    trainer.setTrainConfig(object_names_array=["hololens"], batch_size=4, num_experiments=200, train_from_pretrained_model="pretrained-yolov3.h5")
    trainer.trainModel()

In the first 2 lines, we imported the **DetectionModelTrainer** class and created an instance of it ::
    
    from imageai.Detection.Custom import DetectionModelTrainer

    trainer = DetectionModelTrainer()

Then we called the following functions


* **.setModelTypeAsYOLOv3()** , This function sets the model type of the object detection training instance to the **YOLOv3** model::
    
    trainer.setModelTypeAsYOLOv3()


* **.trainer.setDataDirectory()** , This function is sets the path to your dataset's folder::
    
    trainer.setDataDirectory()

-- *parameter* **data_directory** (required) : This is the path to your dataset folder.



* **.trainer.setTrainConfig()** , This function sets the properties for the training instances::
    
    trainer.setTrainConfig()

-- *parameter* **object_names_array** (required) : This is a list of the names of all the different objects in your dataset.

-- *parameter* **batch_size** (optional) : This is the batch size for the training instance.

-- *parameter* **num_experiments** (required) : Also known as epochs, it is the number of times the network will train on all the training. 

-- *parameter* **train_from_pretrained_model** (optional) : This is used to perform transfer learning by specifying the path to a pre-trained YOLOv3 model


When you run the training code, **ImageAI** will perform the following actions:

- generate a **detection_config.json** in the *dataset_folder/json* folder. Please note that the **JSON** file generated in a training session can only be used with the **detection models** saved in the training session.

- saves the **Tensorboard** report for the training in the *dataet_folder/logs* folder.

- saves new models n the *dataset_folder/models* folder as the training loss reduces.

As the training progresses, the information displayed in the terminal will look similar to the sample below::

    Using TensorFlow backend.
    Generating anchor boxes for training images and annotation...
    Average IOU for 9 anchors: 0.78
    Anchor Boxes generated.
    Detection configuration saved in  hololens/json/detection_config.json
    Training on: 	['hololens']
    Training with Batch Size:  4
    Number of Experiments:  200

    Epoch 1/200
    480/480 [==============================] - 395s 823ms/step - loss: 36.9000 - yolo_layer_1_loss: 3.2970 - yolo_layer_2_loss: 9.4923 - yolo_layer_3_loss: 24.1107 - val_loss: 15.6321 - val_yolo_layer_1_loss: 2.0275 - val_yolo_layer_2_loss: 6.4191 - val_yolo_layer_3_loss: 7.1856
    Epoch 2/200
    480/480 [==============================] - 293s 610ms/step - loss: 11.9330 - yolo_layer_1_loss: 1.3968 - yolo_layer_2_loss: 4.2894 - yolo_layer_3_loss: 6.2468 - val_loss: 7.9868 - val_yolo_layer_1_loss: 1.7054 - val_yolo_layer_2_loss: 2.9156 - val_yolo_layer_3_loss: 3.3657
    Epoch 3/200
    480/480 [==============================] - 293s 610ms/step - loss: 7.1228 - yolo_layer_1_loss: 1.0583 - yolo_layer_2_loss: 2.2863 - yolo_layer_3_loss: 3.7782 - val_loss: 6.4964 - val_yolo_layer_1_loss: 1.1391 - val_yolo_layer_2_loss: 2.2058 - val_yolo_layer_3_loss: 3.1514
    Epoch 4/200
    480/480 [==============================] - 297s 618ms/step - loss: 5.5802 - yolo_layer_1_loss: 0.9742 - yolo_layer_2_loss: 1.8916 - yolo_layer_3_loss: 2.7144 - val_loss: 6.4275 - val_yolo_layer_1_loss: 1.6153 - val_yolo_layer_2_loss: 2.1203 - val_yolo_layer_3_loss: 2.6919
    Epoch 5/200
    480/480 [==============================] - 295s 615ms/step - loss: 4.8717 - yolo_layer_1_loss: 0.7568 - yolo_layer_2_loss: 1.6641 - yolo_layer_3_loss: 2.4508 - val_loss: 6.3723 - val_yolo_layer_1_loss: 1.6434 - val_yolo_layer_2_loss: 2.1188 - val_yolo_layer_3_loss: 2.6101
    Epoch 6/200
    480/480 [==============================] - 300s 624ms/step - loss: 4.7989 - yolo_layer_1_loss: 0.8708 - yolo_layer_2_loss: 1.6683 - yolo_layer_3_loss: 2.2598 - val_loss: 5.8672 - val_yolo_layer_1_loss: 1.2349 - val_yolo_layer_2_loss: 2.0504 - val_yolo_layer_3_loss: 2.5820
    Epoch 7/200



After training is completed, you can evaluate the **mAP** score of your saved models in order to pick the one with the most accurate results. 

To do this, simply run the code below::

    from imageai.Detection.Custom import DetectionModelTrainer

    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory="hololens")
    metrics = trainer.evaluateModel(model_path="hololens/models", json_path="hololens/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)
    print(metrics)


The above code is similar to our training code, except for the line where we called the **evaluateModel()** function. See details on the function below.


* **.trainer.evaluateModel()** , This function allows you to compute and obtain the **mAP** of your saved model(s) based on criterias such as **IoU** and **confidence score** ::
    
    trainer.setTrainConfig()

-- *parameter* **model_path** (required) : This can be the path to a single model or the folder containing your saved models

-- *parameter* **json_path** (required) : This is the **detection_config.json** generated during the training that saved the models.

-- *parameter* **iou_threshold** (optional) : This is used to set the desired **Intersection over Union** for the **mAP** evaluation.

-- *parameter* **object_threshold** (optional) : This is used to set the minimum **confidence score** for the **mAP** evaluation.

-- *parameter* **nms_threshold** (optional) : This is used to set the minimum **Non-maximum Suppression** value for the **mAP** evaluation.


When you run the above code, you get a result similar to the one below::

    [{
        'average_precision': {'hololens': 0.9231334437735249},
        'map': 0.9231334437735249,
        'model_file': 'hololens/models/detection_model-ex-07--loss-4.42.h5',
        'using_iou': 0.5,
        'using_non_maximum_suppression': 0.5,
        'using_object_threshold': 0.3
    },
    {
        'average_precision': {'hololens': 0.9725334437735249},
        'map': 0.97251334437735249,
        'model_file': 'hololens/models/detection_model-ex-10--loss-3.95.h5',
        'using_iou': 0.5,
        'using_non_maximum_suppression': 0.5,
        'using_object_threshold': 0.3
    },
    {
        'average_precision': {'hololens': 0.92041334437735249},
        'map': 0.92041334437735249,
        'model_file': 'hololens/models/detection_model-ex-05--loss-5.26.h5',
        'using_iou': 0.5,
        'using_non_maximum_suppression': 0.5,
        'using_object_threshold': 0.3
    },
    {
        'average_precision': {'hololens': 0.81201334437735249},
        'map': 0.81201334437735249,
        'model_file': 'hololens/models/detection_model-ex-03--loss-6.44.h5',
        'using_iou': 0.5,
        'using_non_maximum_suppression': 0.5,
        'using_object_threshold': 0.3
    },
    {
        'average_precision': {'hololens': 0.94311334437735249},
        'map': 0.94311334437735249,
        'model_file': 'hololens/models/detection_model-ex-18--loss-2.96.h5',
        'using_iou': 0.5,
        'using_non_maximum_suppression': 0.5,
        'using_object_threshold': 0.3
    },
    {
        'average_precision': {'hololens': 0.94041334437735249},
        'map': 0.94041334437735249,
        'model_file': 'hololens/models/detection_model-ex-17--loss-3.10.h5',
        'using_iou': 0.5,
        'using_non_maximum_suppression': 0.5,
        'using_object_threshold': 0.3
    },
    {
        'average_precision': {'hololens': 0.97251334437735249},
        'map': 0.97251334437735249,
        'model_file': 'hololens/models/detection_model-ex-08--loss-4.16.h5',
        'using_iou': 0.5,
        'using_non_maximum_suppression': 0.5,
        'using_object_threshold': 0.3
    }
    ]





**======= imageai.Detection.Custom.CustomObjectDetection =======**

**CustomObjectDetection** class provides very convenient and powerful methods to perform object detection on images and extract each object from the image using your own custom **YOLOv3 model** and the corresponding **detection_config.json** generated during the training. 

To test the custom object detection, you can download a sample custom model we have trained to detect the Hololens headset and its **detection_config.json** file via the links below:


`Hololens Detection Model <https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/hololens-ex-60--loss-2.76.h5>`_

`detection_config.json <https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/detection_config.json>`_

- Sample Image

.. figure:: ../image8.jpg

Once you download the custom object detection model file, you should copy the model file to the your project folder where your **.py** files will be. Then create a python file and give it a name; an example is **FirstCustomDetection.py**. Then write the code below into the python file::

    from imageai.Detection.Custom import CustomObjectDetection

    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath("hololens-ex-60--loss-2.76.h5")
    detector.setJsonPath("detection_config.json")
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image="holo1.jpg", output_image_path="holo1-detected.jpg")
    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])


When you run the code, it will produce a result similar to the one below::

    hololens  :  39.69653248786926  :  [611, 74, 751, 154]
    hololens  :  87.6643180847168  :  [23, 46, 90, 79]
    hololens  :  89.25175070762634  :  [191, 66, 243, 95]
    hololens  :  64.49641585350037  :  [437, 81, 514, 133]
    hololens  :  91.78624749183655  :  [380, 113, 423, 138]


.. figure:: ../image7.jpg


See more details below:


* **.setModelTypeAsYOLOv3()** , This specifies that you are using a trained YOLOv3 model ::
    
    detector.setModelTypeAsYOLOv3()


* **.setModelPath()** , This is used to set the file path to your trained model ::
    
    detector.setModelPath()

-- *parameter* **detection_model_path** (required) : This is path to your model file



* **.setJsonPath()** , This is used to set the file path to your configuration json file ::
    
    detector.setJsonPath()

-- *parameter* **configuration_json** (required) : This is path to *detection_json* file



* **.loadModel()** , This is load the detection model::
    
    detector.loadModel()

* **.detectObjectsFromImage()** , This is the function that performs object detection task after the model as loaded. It can be called many times to detect objects in any number of images. Find example code below::

    detections = detector.detectObjectsFromImage(input_image="image.jpg", output_image_path="imagenew.jpg", minimum_percentage_probability=30)

 -- *parameter* **input_image** (required) : This refers to the path to image file which you want to detect. You can set this parameter to the Numpy array of File stream of any image if you set the paramter **input_type** to "array" or "stream"

 -- *parameter* **output_image_path** (required only if **input_type** = "file" ) :  This refers to the file path to which the detected image will be saved. It is required only if **input_type** = "file" 

 -- *parameter* **minimum_percentage_probability** (optional ) :  This parameter is used to determine the integrity of the detection results. Lowering the value shows more objects while increasing the value ensures objects with the highest accuracy are detected. The default value is 50.

 -- *parameter* **output_type** (optional ) :  This parameter is used to set the format in which the detected image will be produced. The available values are "file" and "array". The default value is "file". If this parameter is set to "array", the function will return a Numpy array of the detected image. See sample below::

     returned_image, detections = detector.detectObjectsFromImage(input_image="image.jpg", output_type="array", minimum_percentage_probability=30)

 -- *parameter* **display_percentage_probability** (optional ) :  This parameter can be used to hide the percentage probability of each object detected in the detected image if set to False. The default values is True.

 -- *parameter* **display_object_name** (optional ) :  This parameter can be used to hide the name of each object detected in the detected image if set to False. The default values is True.

 -- *parameter* **extract_detected_objects** (optional ) :  This parameter can be used to extract and save/return each object detected in an image as a seperate image. The default values is False.

 -- *parameter* **thread_safe** (optional) : This ensures the loaded detection model works across all threads if set to true.

 
 -- *returns* :  The returned values will depend on the parameters parsed into the **detectObjectsFromImage()** function. See the comments and code below
                
        """
            If all required parameters are set and 'output_image_path' is set to a file path you want the detected image to be saved, the function will return:
        
            1. an array of dictionaries, with each dictionary corresponding to the objects 
                detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (list of x1,y1,x2 and y2 coordinates)
        """
        detections = detector.detectObjectsFromImage(input_image="image.jpg", output_image_path="imagenew.jpg", minimum_percentage_probability=30)


        """
            If all required parameters are set and output_type = 'array' ,the function will return

            1. a numpy array of the detected image
            2. an array of dictionaries, with each dictionary corresponding to the objects 
                detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (list of x1,y1,x2 and y2 coordinates)
        """
        returned_image, detections = detector.detectObjectsFromImage(input_image="image.jpg", output_type="array", minimum_percentage_probability=30)


        """
            If extract_detected_objects = True and 'output_image_path' is set to a file path you want
                the detected image to be saved, the function will return:
                1. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (list of x1,y1,x2 and y2 coordinates)
                2. an array of string paths to the image of each object extracted from the image
        """
        detections, extracted_objects = detector.detectObjectsFromImage(input_image="image.jpg", output_image_path="imagenew.jpg", extract_detected_objects=True, minimum_percentage_probability=30)


        """
            If extract_detected_objects = True and output_type = 'array', the the function will return:
                1. a numpy array of the detected image
                2. an array of dictionaries, with each dictionary corresponding to the objects
                    detected in the image. Each dictionary contains the following property:
                    * name (string)
                    * percentage_probability (float)
                    * box_points (list of x1,y1,x2 and y2 coordinates)
                3. an array of numpy arrays of each object detected in the image
        """
        returned_image, detections, extracted_objects = detector.detectObjectsFromImage(input_image="image.jpg", output_type="array", extract_detected_objects=True, minimum_percentage_probability=30)


**======= imageai.Detection.Custom.CustomVideoObjectDetection =======**

**CustomVideoObjectDetection** class provides very convenient and powerful methods to perform object detection on videos and obtain analytical from the video, using your own custom **YOLOv3 model** and the corresponding **detection_config.json** generated during the training. 

To test the custom object detection, you can download a sample custom model we have trained to detect the Hololens headset and its **detection_config.json** file via the links below:


`Hololens Detection Model <https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/hololens-ex-60--loss-2.76.h5>`_

`detection_config.json <https://github.com/OlafenwaMoses/ImageAI/releases/download/essential-v4/detection_config.json>`_

Download a sample video of the Hololens in the link below.

`Sample Hololens Video <https://github.com/OlafenwaMoses/ImageAI/blob/master/data-videos/holo1.mp4>`_


Then run the code below in the video::

    from imageai.Detection.Custom import CustomVideoObjectDetection
    import os

    execution_path = os.getcwd()

    video_detector = CustomVideoObjectDetection()
    video_detector.setModelTypeAsYOLOv3()
    video_detector.setModelPath("hololens-ex-60--loss-2.76.h5")
    video_detector.setJsonPath("detection_config.json")
    video_detector.loadModel()

    video_detector.detectObjectsFromVideo(input_file_path="holo1.mp4",
                                            output_file_path=os.path.join(execution_path, "holo1-detected"),
                                            frames_per_second=30,
                                            minimum_percentage_probability=40,
                                            log_progress=True)


.. figure:: ../customvideodetection.gif

See details on the available functions below


* **.setModelTypeAsYOLOv3()** , This specifies that you are using a trained YOLOv3 model ::
    
    video_detector.setModelTypeAsYOLOv3()


* **.setModelPath()** , This is used to set the file path to your trained model ::
    
    video_detector.setModelPath()

-- *parameter* **detection_model_path** (required) : This is path to your model file



* **.setJsonPath()** , This is used to set the file path to your configuration json file ::
    
    video_detector.setJsonPath()

-- *parameter* **configuration_json** (required) : This is path to *detection_json* file



* **.loadModel()** , This is load the detection model::
    
    video_detector.loadModel()



* **.detectObjectsFromVideo()** , This is the function that performs object detecttion on a video file or video live-feed after the model has been loaded into the instance you created.  Find a full sample code below::

   
 -- *parameter* **input_file_path** (required if you did not set **camera_input**) : This refers to the path to the video file you want to detect. 

 -- *parameter* **output_file_path** (required if you did not set **save_detected_video** = False) : This refers to the path to which the detected video will be saved. By default, this functionsaves video  **.avi** format. 

 -- *parameter* **frames_per_second** (optional , but recommended) : This parameters allows you to set your desired frames per second for the detected video that will be saved. The default value is 20 but we recommend you set the value that suits your video or camera live-feed.

 -- *parameter* **log_progress** (optional) : Setting this parameter to True shows the progress of the video or live-feed as it is detected in the CLI. It will report every frame detected as it progresses. The default value is False.

 -- *parameter* **return_detected_frame** (optional) : This parameter allows you to return the detected frame as a Numpy array at every frame, second and minute of the video detected. The returned Numpy array will be parsed into the respective **per_frame_function**, **per_second_function** and **per_minute_function** (See details below)

 -- *parameter* **camera_input** (optional) : This parameter can be set in replacement of the **input_file_path** if you want to detect objects in the live-feed of a camera. All you need is to load the camera with OpenCV's **VideoCapture()** function and parse the object into this parameter.


 -- *parameter* **minimum_percentage_probability** (optional ) :  This parameter is used to determine the integrity of the detection results. Lowering the value shows more objects while increasing the value ensures objects with the highest accuracy are detected. The default value is 50.

 -- *parameter* **display_percentage_probability** (optional ) :  This parameter can be used to hide the percentage probability of each object detected in the detected video if set to False. The default values is True.

 -- *parameter* **display_object_name** (optional ) :  This parameter can be used to hide the name of each object detected in the detected video if set to False. The default values is True.

 -- *parameter* **save_detected_video** (optional ) :  This parameter can be used to or not to save the detected video or not to save it. It is set to True by default.

 -- *parameter* **per_frame_function** (optional ) :  This parameter allows you to parse in the name of a function you define. Then, for every frame of the video that is detected, the function will be parsed into the parameter will be executed and and analytical data of the video will be parsed into the function. The data returned can be visualized or saved in a NoSQL database for future processing and visualization.
 
    See a sample function for this parameter below::
        """
        This parameter allows you to parse in a function you will want to execute after
        each frame of the video is detected. If this parameter is set to a function, after every video
        frame is detected, the function will be executed with the following values parsed into it:
        -- position number of the frame
        -- an array of dictinaries, with each dictinary corresponding to each object detected.
            Each dictionary contains 'name', 'percentage_probability' and 'box_points'
        -- a dictionary with with keys being the name of each unique objects and value
           are the number of instances of each of the objects present
        -- If return_detected_frame is set to True, the numpy array of the detected frame will be parsed
            as the fourth value into the function
        """

        def forFrame(frame_number, output_array, output_count):
            print("FOR FRAME " , frame_number)
            print("Output for each object : ", output_array)
            print("Output count for unique objects : ", output_count)
            print("------------END OF A FRAME --------------")

 -- *parameter* **per_second_function** (optional ) :  This parameter allows you to parse in the name of a function you define. Then, for every second of the video that is detected, the function will be parsed into the parameter will be executed and analytical data of the video will be parsed into the function. The data returned can be visualized or saved in a NoSQL database for future processing and visualization.
 
    See a sample function for this parameter below::
        """
        This parameter allows you to parse in a function you will want to execute after
        each second of the video is detected. If this parameter is set to a function, after every second of a video
        is detected, the function will be executed with the following values parsed into it:
        -- position number of the second
        -- an array of dictionaries whose keys are position number of each frame present in the last second , and the value for each key is the array for each frame that contains the dictionaries for each object detected in the frame

        -- an array of dictionaries, with each dictionary corresponding to each frame in the past second, and the keys of each dictionary are the name of the number of unique objects detected in each frame, and the key values are the number of instances of the objects found in the frame

        -- a dictionary with its keys being the name of each unique object detected throughout the past second, and the key values are the average number of instances of the object found in all the frames contained in the past second

        -- If return_detected_frame is set to True, the numpy array of the detected frame will be parsed as the fifth value into the function
        """

        def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
            print("SECOND : ", second_number)
            print("Array for the outputs of each frame ", output_arrays)
            print("Array for output count for unique objects in each frame : ", count_arrays)
            print("Output average count for unique objects in the last second: ", average_output_count)
            print("------------END OF A SECOND --------------")

 -- *parameter* **per_minute_function** (optional ) :  This parameter allows you to parse in the name of a function you define. Then, for every frame of the video that is detected, the function which was parsed into the parameter will be executed and analytical data of the video  will be parsed into the function. The data returned has the same nature as the **per_second_function** ; the difference is that it covers all the frames in the past 1 minute of the video.
    
    See a sample function for this parameter below::

        def forMinute(minute_number, output_arrays, count_arrays, average_output_count):
            print("MINUTE : ", minute_number)
            print("Array for the outputs of each frame ", output_arrays)
            print("Array for output count for unique objects in each frame : ", count_arrays)
            print("Output average count for unique objects in the last minute: ", average_output_count)
            print("------------END OF A MINUTE --------------")
 


 -- *parameter* **video_complete_function** (optional ) :  This parameter allows you to parse in the name of a function you define. Once all the frames in the video is fully detected, the function will was parsed into the parameter will be executed and analytical data of the video  will be parsed into the function. The data returned has the same nature as the **per_second_function** and **per_minute_function** ; the differences are that no index will be returned and it covers all the frames in the entire video. 
 
    See a sample funtion for this parameter below::

        def forFull(output_arrays, count_arrays, average_output_count):
            print("Array for the outputs of each frame ", output_arrays)
            print("Array for output count for unique objects in each frame : ", count_arrays)
            print("Output average count for unique objects in the entire video: ", average_output_count)
            print("------------END OF THE VIDEO --------------")

 -- *parameter* **detection_timeout** (optional) :  This function allows you to state the number of seconds of a video that should be detected after which the detection function stop processing the video.
 
    


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   

