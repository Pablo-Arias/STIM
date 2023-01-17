from feat import Fex
from PIL import Image
import matplotlib.pyplot as plt
import feat

def plot_landmarks_on_images():	
	pass

def plot_features_on_image(source, target, detector,landmark_size =15, axis_off=True, draw_landmarks=True, draw_face_detection=True, draw_facepose=True):
	"""
	| Extract and plot action units in an image (useful when doing presentations)
	| Input:
	|	source : Array like with image file names
	|	target : Array like with target image file names
	|	
	| Example:
	| 
	|	from feat import Detector
	|	#define_detector
	|	detector  = Detector(face_model     = "retinaface"
	| 	                   , landmark_model = "mobilenet"
	| 	                   , au_model       = "rf"
	| 	                   , emotion_model  = "resmasknet"
	| 	                  )
	|	plot_landmarks_on_image(source, target, detector,landmark_size =15)
	| 
	"""

	from matplotlib.patches import Rectangle
	image_prediction = detector.detect_image(source)

	
	#draw image
	f, ax = plt.subplots()
	im = Image.open(source)
	ax.imshow(im);

	#draw landmarks
	if draw_landmarks:
		x = image_prediction.landmark_x().values[0]
		y = image_prediction.landmark_y().values[0]

		#plot
		plt.scatter(x, y, s=landmark_size, color="blue")

	#draw face detection
	if draw_face_detection:
		facebox = image_prediction.facebox().values[0]

		#Define face rectangle
		rect = Rectangle(
		    (facebox[0], facebox[1]),
		    facebox[2],
		    facebox[3],
		    linewidth=2,
		    edgecolor="cyan",
		    fill=False,
		)

		#Plot image
		im = Image.open(source)
		ax.imshow(im);
		ax.add_patch(rect)


	if draw_facepose:
		facebox = image_prediction.facebox().values[0]
		feat.plotting.draw_facepose(pose=image_prediction.facepose().values[0], facebox=facebox, ax=ax)

	if axis_off:
		plt.axis('off')		

	#savefig
	plt.savefig(target)		


