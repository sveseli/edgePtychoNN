# PvaPy Streaming Framework Implementation of PtychoNN

PtychoNN code does real-time inference using NVIDIA Jetson AGX Xavier Developer kit on the diffraction patterns streamed out from the X-ray detector. The images are streamed out to the Jetson as PVA stream and fed into the inference engine using TensorRT Python API. The embedded GPU system will then perform the inference and sends back the inference outputs as PVA stream to the user interface for viewing. The inference is done in batch processing with a batch size of 8 during the demonstration. 

The original code has been converted to use PvaPy Streaming Framework
using the new [inferPtychoNNImageProcessor.py] processor file and
[inferPtychoNNEngine.py] which replaces existing [inferPtychoNN.py]
file. Other files related to streaming ([pvaClient.py] and [adSimServer.py]) 
are also not needed.

Steps to run the code. 

1. In one terminal run the adSimServer.py, which will simulate a detector with either a random 128x128 images or from a presaved scan file (scan810.npy) 
   python3 adSimServer.py -if diff_scan_810.npy -fps 10
   
   fps is the frame rate and the maximum possible frame rate with real-time feedback is 2000.
   
2. Open another terminal and run the main-batch-test.py 
