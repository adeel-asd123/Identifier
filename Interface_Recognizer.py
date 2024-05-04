from tkinter import filedialog
import tkinter
from tkinter import messagebox
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from imutils.video import FPS
from imutils.video import VideoStream
import time
Main= tkinter.Tk()
def trainer():
	messagebox.showinfo("Check", "Make sure you have your dataset in a folder, which is named what your identifing, inside the folder named dataset")
	print("Loading Face Detector...")
	protoPath = r"face_dectection_model\deploy.prototxt"
	modelPath = r"face_dectection_model\res10_300x300_ssd_iter_140000.caffemodel"
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

	# load serialized face embedding model
	print("Loading Face Recognizer...")
	embedder = cv2.dnn.readNetFromTorch(r"openface_nn4.small2.v1.t7")

	# grab the paths to the input images in our dataset
	print("Quantifying Faces...")
	imagePaths = list(paths.list_images("dataset"))

	# initialize our lists of extracted facial embeddings and corresponding people names
	knownEmbeddings = []
	knownNames = []

	# initialize the total number of faces processed
	total = 0
	
# loop over the image paths
	for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
		if (i%50 == 0):
			print("Processing image {}/{}".format(i, len(imagePaths)))
		name = imagePath.split(os.path.sep)[-2]

	# load the image, resize it to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
		image = cv2.imread(imagePath)
		image = imutils.resize(image, width=600)
		(h, w) = image.shape[:2]

	# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(image, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()

	# ensure at least one face was found
		if len(detections) > 0:
			# we're making the assumption that each image has only ONE face, so find the bounding box with the largest probability
			i = np.argmax(detections[0, 0, :, 2])
			confidence = detections[0, 0, i, 2]

		# ensure that the detection with the largest probability also means our minimum probability test (thus helping filter out weak detections)
			if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI and grab the ROI dimensions
				face = image[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

			# construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

			# add the name of the person + corresponding face embedding to their respective lists
				knownNames.append(name)
				knownEmbeddings.append(vec.flatten())
				total += 1

# dump the facial embeddings + names to disk
	print("[INFO] serializing {} encodings...".format(total))
	data = {"embeddings": knownEmbeddings, "names": knownNames}
	f = open("output/embeddings.pickle", "wb")
	f.write(pickle.dumps(data))
	f.close()
# load the face embeddings
	print("[INFO] loading face embeddings...")
	data = pickle.loads(open("output/embeddings.pickle", "rb").read())

    # encode the labels
	print("[INFO] encoding labels...")
	le = LabelEncoder()
	labels = le.fit_transform(data["names"])

    # train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
	print("[INFO] training model...")
	recognizer = SVC(C=1.0, kernel="linear", probability=True)
	recognizer.fit(data["embeddings"], labels)

    # write the actual face recognition model to disk
	f1 = open("output/recognizer", "wb")
	f1.write(pickle.dumps(recognizer))
	f1.close()

    # write the label encoder to disk
	f1 = open("output/le.pickle", "wb")
	f1.write(pickle.dumps(le))
	f1.close()
photofiles = r'*.png  *.jpg  .jpeg'
videofiles = r'*.mp4  *.mov'
Main.title('Dataset Recognizer')
  
# Set Main size
Main.geometry("700x200")
  
#Set Main background color
Main.config(background = "white")
def Image_Recognizer(file):
	
	protoPath = r"face_dectection_model\deploy.prototxt"
	modelPath =r"face_dectection_model\res10_300x300_ssd_iter_140000.caffemodel"
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
	
	embedder = cv2.dnn.readNetFromTorch(r"openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
	recognizer = pickle.loads(open(r'output\recognizer', "rb").read())
	le = pickle.loads(open('output/le.pickle', "rb").read())

# load the image, resize it to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
	image = cv2.imread(file)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

# apply OpenCV's deep learning-based face detector to localize faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

		# extract the face ROI
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

		# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

		# construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
				(0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

		# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

		# draw the bounding box of the face along with the associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(image, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# show the output image
	cv2.imshow("Image", image)
	cv2.waitKey(0)
def live_Recognizer():
	print("Loading Face Detector...")
	protoPath = r"face_dectection_model\deploy.prototxt"
	modelPath = r"face_dectection_model\res10_300x300_ssd_iter_140000.caffemodel"
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

	# load serialized face embedding model
	print("Loading Face Recognizer...")
	embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
	recognizer = pickle.loads(open("output/recognizer", "rb").read())
	le = pickle.loads(open("output/le.pickle", "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
	print("Starting Video Stream...")
	vs = VideoStream(src=0).start()

# start the FPS throughput estimator
	fps = FPS().start()

# loop over frames from the video file stream
	while True:
		# grab the frame from the threaded video stream
		frame = vs.read()

	# resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
		frame = imutils.resize(frame, width=600)
		(h, w) = frame.shape[:2]

	# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()

	# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the prediction
			confidence = detections[0, 0, i, 2]

		# filter out weak detections
			if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
				face = frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

			# construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

			# perform classification to recognize the face
				preds = recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = le.classes_[j]

			# draw the bounding box of the face along with the associated probability
				text = "{}: {:.2f}%".format(name, proba * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 0, 255), 2)
				cv2.putText(frame, text, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# update the FPS counter
		fps.update()
    
	# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

# stop the timer and display FPS information
	fps.stop()
	print("Elasped time: {:.2f}".format(fps.elapsed()))
	print("Approx. FPS: {:.2f}".format(fps.fps()))

# cleanup
	cv2.destroyAllWindows()
	vs.stop()
label_file_explorer = tkinter.Label(Main, 
                            text = "Select file or video",
                            width = 100, height = 4, 
                            fg = "blue")
  
def filechooserandintialzer():
	global filename
	Which_One = messagebox.askquestion("Photo?","If No, a live window will be launched")
	if Which_One == "no":
		live_Recognizer()
	else:
		filename = filedialog.askopenfilename(initialdir = "/",
                        title = "Select a File",  
						filetypes = [("Photos",photofiles),('Videos',videofiles)])
		label_file_explorer.config(text=filename)
		Image_Recognizer(filename)
button_start = tkinter.Button(Main, 
                        text = "Start Recognizing",
                        command = filechooserandintialzer) 
button_Train = tkinter.Button(Main, text="Train, DO THIS FIRST", command= trainer)

# Grid method is chosen for placing
# the widgets at respective positions 
# in a table like structure by
# specifying rows and columns
label_file_explorer.grid(column = 1, row = 1)
  
button_start.grid(column = 1, row = 2)

button_Train.grid(column=1, row=3)
 

# Let the Main wait for any events
Main.mainloop()