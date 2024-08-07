# PvaPy Streaming Framework Implementation of PtychoNN

PtychoNN code does real-time inference using NVIDIA Jetson AGX Xavier Developer kit on the diffraction patterns streamed out from the X-ray detector. The images are streamed out to the Jetson as PVA stream and fed into the inference engine using TensorRT Python API. The embedded GPU system will then perform the inference and sends back the inference outputs as PVA stream to the user interface for viewing. The inference is done in batch processing with a batch size of 8 during the demonstration. 

The original code has been converted to use PvaPy Streaming Framework
using the new [inferPtychoNNImageProcessor.py](inferPtychoNNImageProcessor.py) processor file and
[inferPtychoNNEngine.py](inferPtychoNNEngine.py) which replaces existing [inferPtychoNN.py](inferPtychoNN.py)
file. Other files related to streaming ([pvaClient.py](pvaClient.py) and [adSimServer.py](adSimServer.py)) 
are also not needed, which reduces the code base by about 40%.

## Environment Setup

* Prepare your environment by installing conda and the required
  packages. Note that the code was developed using Tensor RT 8.x, and
  would require changes for newer version (see [Nvidia TensorRT Migration Guide]
  guide](https://docs.nvidia.com/deeplearning/tensorrt/migration-guide/index.html)).

```sh
$ conda create -n PtychoNN
$ conda activate PtychoNN
(PtychoNN) $ conda install python=3.9
(PtychoNN) $ pip install nvidia-pyindex
(PtychoNN) $ pip install nvidia-tensorrt==8.4.1.5
(PtychoNN) $ pip install torch-summary
(PtychoNN) $ conda install -c conda-forge pycuda=2022.1
(PtychoNN) $ conda install -c pytorch pytorch torchvision
(PtychoNN) $ conda install -c epics pvapy
(PtychoNN) $ pip install c2dataviewer # optional, for viewing images over EPICS PVA channels
```

* Checkout git repo:

```sh
(PtychoNN) $ git clone https://github.com/sveseli/edgePtychoNN.git
```
## Demo Steps

Any steps shown below that use PtychoNN code or example files should
be executed from the top level folder of the above git repository after
it has been cloned.

Note that the frame rate that the system can handle without loosing any
frames will entirely depend on the machine specs. On the machine on
which this code was originally developed (Intel Xeon(R) Gold 6342 CPU @ 2.80GHz, with 96 logical CPU cores, 2TB RAM, dual NVIDIA RTX A6000 GPU), a single
instance of inference engine could handle stream of 128x128 int16 images at rates of up to 2kHZ. Using PvaPy Streaming Framework to distribute data to 4 consumer processes (each running single inference engine), the code was capable of keeping up with frame rates of 8kHz.

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

Note that it may take 30 seconds or so before the inference engine is fully
initialized. The software can receive images, but will not start processing
them until the initialization stage is complete.

* In Terminal 2, start AD sim server:

```sh
(PtychoNN) $ pvapy-ad-sim-server -cn ad:image -nx 128 -ny 128 -dt int16 -fps 1000 -rt 60 -rp 1000
```

The above will stream random images on the channel 'ad:image' at a rate
of 1000 frames per second, for 60 seconds.
Alternatively, one can stream included example scan in the 'data' folder:

```sh
(PtychoNN) $ pvapy-ad-sim-server -cn ad:image -if data/scan674.npy -fps 1000 -rt 60 -rp 1000
```

* In Terminal 3, observe stats:

```sh
$ watch -d 'nvidia-smi | tail -12'
```

* In Terminal 4, one can view both the raw images on the 'ad:image'
  channel, as well as the processed frames on the 'inference:N:output',
  where N is the consumer number (consumer numbering starts with 1).

```sh
$ c2dv --app image --pv ad:image &
$ c2dv --app image --pv inference:1:output &
```

