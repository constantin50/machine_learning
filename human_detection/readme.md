## Human Detection Model

This model gives prediction that there is a human on a picture, it can be used to process video stream from CCTV cameras.
The basis of a model is a neural network with LeNet-based architecture. Model is avaible as colab notebook, as pytorch model saved with state_dict() method and as GUI application.

## Get started

### Google Colab

Just run human_detection.ipynb file

```
PATH1 # path to model_32
PATH2 # path to image (png/jpg/bmp) 

model = LeNet()
model.load_state_dict(torch.load(PATH1, map_location=torch.device('cpu')))
model.eval()

result = predict(model, PATH2) 

print(result)
```

### Load pytorch model




## Dataset 

Dataset contains CCTV footage images(as indoor as outdoor), 552 of them with humans and 462 without them. Images 

![image](https://github.com/constantin50/machine_learning/blob/master/human_detection/thumbnail.png)
