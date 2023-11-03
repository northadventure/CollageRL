# Neural Collage Transfer: Artistic Reconstruction via Material Manipulation (ICCV 2023)
[Ganghun Lee](https://scholar.google.com/citations?hl=ko&authuser=1&user=wN-2e1MAAAAJ), Minji Kim, Yunsu Lee, [Minsu Lee](https://scholar.google.com/citations?hl=ko&authuser=1&user=75_DkUwAAAAJ), and [Byoung-Tak Zhang](https://scholar.google.com/citations?hl=ko&authuser=1&user=sYTUOu8AAAAJ)

## Description
An official implementation of the paper [Neural Collage Transfer: Artistic Reconstruction via Material Manipulation](https://openaccess.thecvf.com/content/ICCV2023/html/Lee_Neural_Collage_Transfer_Artistic_Reconstruction_via_Material_Manipulation_ICCV_2023_paper.html).

### Examples
<img width=500 src="resource/actress.gif"><img width=500 src="resource/colorface.gif">

## Requirements
- Python 3.8.5 (Conda)
- PyTorch 1.11.0

We recommend using the following instruction after making a new Python 3.8.5 environment:<br>
```$ pip install -r requirements.txt```


## Inference
You can find ```infer.sh``` for testing your own image.

### Goal image

The ```goal``` image should be placed in ```samples/goal/```.<br>ex) ```samples/goal/boat.jpg```
### Material images

The ```materials``` are a set of images, so please make your own folder (e.g., newspaper/) containing all your material images.<br>Then move the folder to the directory ```samples/materials/```.<br>ex) ```samples/materials/newspaper/```

To make it quick, you can download a prepared set of newspapers from [here](https://archive.ics.uci.edu/dataset/306/newspaper+and+magazine+images+segmentation+dataset).<br>(Vilkin,Aleksey and Safonov,Ilia. (2014). Newspaper and magazine images segmentation dataset. UCI Machine Learning Repository. [https://doi.org/10.24432/C5N60V](https://doi.org/10.24432/C5N60V).)<br>There would be some kinds of files, but we only need the ```.jpg```s (please delete the other files). 

### Instruction

Please make sure to set your goal/material path  in ```infer.sh```.<br>```GOAL_PATH='samples/goals/your_own_goal.jpg'``` (not necessarily .jpg extension)<br>```SOURCE_DIR='samples/materials/your_own_material_folder'```<br>

Now you can run the code.<br>```$ bash infer.sh```<br>It will take some time, and the results will be saved at ```samples/results/```.

### Configuration
<!-- If you want to adjust something, please set the values in ```infer.sh```. -->
- ```GOAL_RESOLUTION``` - result image resolution
- ```GOAL_RESOLUTION_FIT``` - fit the resolution as (horizontal | vertical | square)
- ```SOURCE_RESOLUTION_RATIO``` - material image casting size (0-1)
- ```SOURCE_LOAD_LIMIT``` - max num of material images to load (prevent RAM overloaded)
- ```SOURCE_SAMPLE_SIZE``` - num of material images agent will see at each step
- ```MIN_SOURCE_COMPLEXITY``` - minimum allowed complexity for materials (prevent using too simple ones) (>=0)
- ```SCALE_ORDER``` - scale sequence for multi-scale collage
- ```NUM_CYCLES``` - num of steps for each sliding window
- ```WINDOW_RATIO``` - stride ratio of sliding window (0-1) (0.5 for stride = window_size x 0.5)
- ```MIN_SCRAP_SIZE``` - the minimum allowed scrap size (prevent too small scraps) (0-1)
- ```SENSITIVITY``` - complexity-sensitivity value for multi-scale collage
- ```FIXED_T``` - fixed value of t_channel for multi-scale collage
- ```FPS``` - fps for result video

You can also toggle the following options:
- ```skip_negative_reward``` - Whether to undo actions that led to a negative MSE reward
- ```paper_like``` - Whether to use the torn paper effect
- ```disallow_duplicate``` - Whether to disallow duplicate usage of materials

We recommend trying adjusting ```SENSITIVITY``` first, in a range of about 1-5.

## Training

### Dataset
Goals and materials should be prepared for training.

#### Goal set
This code supports the following datasets for goals:
- [ImageNet](https://www.image-net.org/) (2012)
- [MNIST]()
- [Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)
- [Scene](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

#### Material set
This code properly supports [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/) for training only.

Please make your datasets be placed in the same data directory.<br>As an example, you can see our example tree of ```~/Datasets```.
```
Datasets/
├── dtd
│   ├── images
│   ├── imdb
│   └── labels
├── flowers-102
│   ├── imagelabels.mat
│   ├── jpg
│   └── setid.mat
├── imagenet
│   ├── meta.bin
│   ├── train
│   └── val
├── IntelScene
│   ├── train
│   └── val
└── MNIST
    └── raw
```

Then set ```--data_path``` in ```train.sh``` to your data directory.<br>ex) ```--data_path ~/Datasets```

### Wandb
Before training, please set up and log in to your [wandb](https://wandb.ai/) account for logging.

### Instruction

Set ```--goal``` in ```train.sh``` to right name (imagenet | mnist | flower | scene).<br> Tip: ```imagenet``` is for general use.

```--source``` means ```material```, and it basically supports ```dtd``` only.<br>
 But you can use other materials for specific goal-material cases: (imagenet-imagenet, mnist-mnist, flower-flower, scene-scene).

Now just run the code to train:<br>```$ bash train.sh```

The progress and result will be saved at ```outputs/```.<br>If your RAM get overloaded, you can decrease the replay memory size ```--replay_size```.

### Renderer (Shaper)
To make the rendering process differentiable, we implemented and pretrained ```shaper``` network as in ```shaper/shaper_training.ipynb```.<br>We also used [Kornia](https://github.com/kornia/kornia) library for differentiable image translation.

## Citation
If you find this work useful, please cite the paper as follows:
```
@inproceedings{lee2023neural,
  title={Neural Collage Transfer: Artistic Reconstruction via Material Manipulation},
  author={Lee, Ganghun and Kim, Minji and Lee, Yunsu and Lee, Minsu and Zhang, Byoung-Tak},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2394--2405},
  year={2023}
}
```

## Acknowledgements
Many thanks to the authors of [Learning to Paint](https://github.com/megvii-research/ICCV2019-LearningToPaint) for inspiring this work. They also inspired our other work [From Scratch to Sketch](https://arxiv.org/abs/2208.04833).<br>
We also appreciate the contributors of [Kornia](https://github.com/kornia/kornia) for providing useful differentiable image processing operators.
