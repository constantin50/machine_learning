import tkinter
from tkinter import filedialog
from tkinter import *
import cv2
import matplotlib.pyplot as plt 
import PIL
from PIL import Image, ImageOps, ImageTk
from torchvision import transforms
import math
class App:
    def __init__(self, window, window_title, model, delay=15, plot=False):
        self.window = window
        self.window.title(window_title)
        self.model = model
        self.status = 0
        self.history = list()
        self.plot = plot

        self.indic = Label(self.window, text="false")
        self.indic.config(font=("Courier", 24),)
        self.indic.pack()

        b1 = Button(self.window, text="Select video", command=self.select)
        b1.pack()

        b2 = Button(self.window, text="Web-camera", command=self.webcamera)
        b2.pack()
 
        self.window.mainloop()

    def select(self):
        self.video_source = tkinter.filedialog.askopenfilename()
        self.video = MyVideoCapture(self.video_source)
        self.canvas = tkinter.Canvas(self.window, width = self.video.width, height = self.video.height+150)
        self.canvas.pack()
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

    def webcamera(self):
        self.video = MyVideoCapture(0)
        self.canvas = tkinter.Canvas(self.window, width = self.video.width, height = self.video.height+150)
        self.canvas.pack()
        self.delay = 15
        self.update()        

    def predict(self, image):
        img = image
        img = img.resize((64, 64), Image.ANTIALIAS)
        trans = transforms.ToTensor()
        img = trans(img)
        input_data = img.unsqueeze(0)
        result = self.model.forward(input_data)
        return float(result[0][1].data/(math.sqrt(float((result[0][0].data)*(result[0][0].data)) + float((result[0][1].data)*(result[0][1].data)))))
	 
 
    def update(self):
        # Get a frame from the video source
        ret, frame = self.video.get_frame()
 		
        if ret:
            res = self.predict(PIL.Image.fromarray(frame))
            print(res)
            if (self.plot==True):
                self.history.append(res)
                plt.plot(self.history, color='c')
                plt.pause(0.001)
            if (res > 0):
                self.indic.config(text="true")
            else:
               self.indic.config(text="false")

            indic = Label(self.window, text=str(self.predict(PIL.Image.fromarray(frame))))


            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
 
        self.window.after(self.delay, self.update)

class MyVideoCapture:
	def __init__(self, video_source):

		# Open the video source
		self.video = cv2.VideoCapture(video_source)
		if not self.video.isOpened():
			raise ValueError("Unable to open video source", video_source)
 		

       # Get video source width and height
		self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
		self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
	def get_frame(self):
		if self.video.isOpened():
			ret, frame = self.video.read()
			if ret: return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
			else: return (ret, None)
		else: return (ret, None)
 
   # Release the video source when the object is destroyed
	def __del__(self):
		if self.video.isOpened():
			self.video.release()