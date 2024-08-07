# PvaPy Streaming Framework Implementation of PtychoNN

PtychoNN code does real-time inference using NVIDIA Jetson AGX Xavier Developer kit on the diffraction patterns streamed out from the X-ray detector. The images are streamed out to the Jetson as PVA stream and fed into the inference engine using TensorRT Python API. The embedded GPU system will then perform the inference and sends back the inference outputs as PVA stream to the user interface for viewing. The inference is done in batch processing with a batch size of 8 during the demonstration. 

The original code has been converted to use PvaPy Streaming Framework
using the new [inferPtychoNNImageProcessor.py](inferPtychoNNImageProcessor.py) processor file and
[inferPtychoNNEngine.py](inferPtychoNNEngine.py) which replaces existing [inferPtychoNN.py](inferPtychoNN.py)
file. Other files related to streaming ([pvaClient.py](pvaClient.py) and [adSimServer.py](adSimServer.py)) 
are also not needed, which reduces the code base by about 40%.

## Environment Setup

* Prepare your environment by installing conda and the required packages:

```sh
$ conda create -n PtychoNN
$ conda activate PtychoNN
(PtychoNN) $ conda install python=3.9
(PtychoNN) $ pip install nvidia-pyindex
(PtychoNN) $ pip install nvidia-tensorrt
(PtychoNN) $ pip install torch-summary
(PtychoNN) $ conda install -c conda-forge pycuda
(PtychoNN) $ conda install -c pytorch pytorch torchvision
(PtychoNN) $ conda install -c epics pvapy
```

* Checkout git repo:

```sh
(PtychoNN) $ git clone https://github.com/sveseli/edgePtychoNN.git
```
## Demo Steps

* In Terminal 1, start processing consumers:

```sh
(PtychoNN) $ cd edgePtychoNN
(PtychoNN) $ export PYTHONPATH=$PWD
(PtychoNN) $ pvapy-hpc-consumer --input-channel ad:image --control-channel inference:*:control --status-channel inference:*:status --output-channel inference:*:output --processor-file inferPtychoNNImageProcessor.py --processor-class InferPtychoNNImageProcessor --report-period 10 --server-queue-size 100 --n-consumers 4 --distributor-updates 8
```

The above command should start 4 instances of inference engine running
on machine's GPUs. Each application instance will be receiving images from the
'ad:image' PVA channel in the batches of 8: the first one will receive
frames [1-8,33-40,...], the second one will receive frames [9-16,41-48,...], the third one will receive frames [17-24,49-56,...], and the fourth one will be getting frames [25-32,57-64,...]. This is controlled by the '--distributor-updates' options in the above command.

* In Terminal 2, start AD sim server:

```sh
(PtychoNN) $ pvapy-ad-sim-server -cn ad:image -nx 128 -ny 128 -dt int16 -fps 8000 -rt 60 -rp 8000
```

The above will stream random images on the channel 'ad:image'.
Alternatively, stream included example scan from the 'data' folder:

```sh
(PtychoNN) $ pvapy-ad-sim-server -cn ad:image -if data/scan674.npy -fps 8000 -rt 60 -rp 8000
```

The above will stream random images on the channel 'ad:image'.


* In Terminal 3, observe stats:

```sh
$ watch -d 'nvidia-smi | tail -12'
```
