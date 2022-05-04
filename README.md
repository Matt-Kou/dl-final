# NYU CSCI2022 Spring Deep Learning Final Project

## Group Members:

Yifeng Kou yk1962
Yuanzhe Liu yl9539
Jingnan Zhu jz5313
Group 11



# Goal:

Use unsupervised learning technique to do object detection.

## Solution:

We use **Dino **to pretrain the resnet-50 backbone and then use faster rcnn to train the model for object detection.

(We tried to use **SWAV** but Dino performs way better than SWAV. ) 

## Usage:

### Dino (to get the backbone):

 **Yifeng Kou**: 

### To get the teacher backbone:

**Yifeng Kou**: 

### Faster RCNN:

Save the model from DINO training and the teacher backbone.

**demo_dino.py** assumes the model is in the **same folder** as demo_dino.py

In **demo_dino.py**:

1. go to the `def get_model(num_classes):` function
2. change the name inside `backbone = torch.load('resnet-full-nofc')` (i.e. replace `resnet-full-nofc`)
3. create a fold named **model** in the current folder (the **same folder** of **demo_dino.py**)
4. In the `def main():` function, in the loop `for epoch in range(num_epochs):`, go to `torch.save(model, os.path.join('model', 'dino_epoch_b2_1gpu-{}'.format(epoch)))`, feel free to change the name of the model.
5. run `python demo_dino.py`

Right now, The model name is **dino_epoch_b2_1gpu-14**, where we pretrain the resnet-50 backbone on dino, and then use fastrcnn in demo to train the backbone with batchsize 2 and only 14 epoch.

## Evaluation

To run the evaluation, just do:
python evaluate.py

or

sbatch eval.slurm

if you want to check the result, it's in eval_75629.out

evaluation results:

IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.060
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.104
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.063
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.002
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.023
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.070
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.157
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.177
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.177
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.003
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.070
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.203