## Human Detection Model

This model gives prediction that there is a human on a picture, it can be used to process video stream from CCTV cameras.
The basis of a model is a neural network with LeNet-based architecture. The model was trained on the dataset of 64x64 images. Model is avaible as colab notebook, as pytorch model saved with state_dict() method and as GUI application.

## Validation accuracy

Mean validation accuracy is 0.79%

![validation accuracy](https://github.com/constantin50/machine_learning/blob/master/human_detection/eyE35WNwJzw.jpg)


## Get started

### Google Colab (GPU)

Just run human_detection.ipynb file

```
PATH1 # path to model_64
PATH2 # path to image (png/jpg/bmp) 

model = LeNet()
model.load_state_dict(torch.load(PATH1, map_location=torch.device('cpu')))
model.eval()

result = predict(model, PATH2) 

print(result)
```

### GUI via Python idle (CPU)

```
import torch
import tkinter
import LeNet
import Human_Detection as HD

PATH1 = 'C:\projects\human detection\model_64' # path to model_64 (can be changed)
model = LeNet.LeNet() # define model from LeNet 
model.load_state_dict(torch.load(PATH1, map_location=torch.device('cpu'))) # load trained weights from model_64
<All keys matched successfully>
model.eval() # switch model into prediction mode
...
HD.App(tkinter.Tk(), delay=15, window_title="human_detection", model=model, plot=True) # if you do not want to display real-time probability plot switch the parameter to 'False'
```

![how it looks](https://github.com/constantin50/machine_learning/blob/master/human_detection/demo.gif)

in a Python window you see probability of that a current frame contains a human, a plot window displays history of probabilities.  

## Dataset 

Dataset contains CCTV footage 256x256 images(as indoor as outdoor), 684 of them with humans and 470 without them.

You can download it from [Kaggle](https://www.kaggle.com/constantinwerner/human-detection-dataset)

![image](https://github.com/constantin50/machine_learning/blob/master/human_detection/thumbnail.png)
