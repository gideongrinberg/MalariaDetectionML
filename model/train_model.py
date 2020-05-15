# IMPORTANT: This won't run on repl.it. This is a GPU model, and repl.it doesn't use GPUs. There is a conflict with the CPU fastai used in the server. Even with the proper dependences, this would take at least 3 hours here. Instead, you can view or run it at this Kaggle notebook: https://bit.ly/3bxJIaw. This file outputs the pkl model that is used by the server.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastai import *
from fastai.vision import *

torch.cuda.is_available()

image_path = os.path.abspath('../input/cell-images-for-detecting-malaria/cell_images')
data = ImageDataBunch.from_folder(image_path, 
                                  train=".",
                                  valid_pct=0.2, 
                                  ds_tfms=get_transforms(flip_vert=True, max_warp=0),
                                  size=224,bs=64, 
                                  num_workers=0).normalize(imagenet_stats)
data.show_batch(rows=5)
print(f'Classes: \n {data.classes}')

learn = cnn_learner(data, models.resnet50, metrics=[accuracy, error_rate], model_dir="models/temp/model")

learn.fit_one_cycle(5, slice(1e-5, learn_rate/5))

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))

learn.save('stage-2-rn50')