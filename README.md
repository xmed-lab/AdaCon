# AdaCon: Adaptive Contrast for Image Regression in Computer-Aided Disease Assessment

<p align="center">
  <img src="https://github.com/XMed-Lab/AdaCon/raw/main/echonet/docs/framework_chart.PNG" width="300">
</p>

This is the implementation of AdaCon on the EchoNet-Dynamic Dataset for the paper ["AdaCon: Adaptive Contrast for Image Regression in Computer-Aided Disease Assessment"](http://arxiv.org/abs/2112.11700) (IEEE TMI).

### Data

Researchers can request the EchoNet-Dynamic dataset at https://echonet.github.io/dynamic/ and set the directory path in the configuration file, `echonet.cfg`.


### Environment

It is recommended to use PyTorch `conda` environments for running the program. A requirements file has been included. 

### Training

The code must first be installed by running 
    
    pip install --user .

under the `adacon` directory. To train the model from scratch, run:

```
echonet video --frames=32 --model_name=r2plus1d_18 --period=2 --batch_size=20 --run_test --output=training_output
```


### Testing

A trained version of the model can be downloaded from https://hkustconnect-my.sharepoint.com/:u:/g/personal/wdaiaj_connect_ust_hk/EXu95kAzcitGibTOWxwSmDEBXkPIZsyYSt1dXurQDpsE3g?e=lwS3mO. 

Inference with the trained model can be run using

```
echonet video --frames=32 --model_name=r2plus1d_18 --period=2 --batch_size=20 --run_test --output=training_output --weights=<PATH TO MODEL> --num_epochs=0
```