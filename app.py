# Core Pkgs

from json import load
import streamlit as st 
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
from io import BytesIO
st.title("Face Detection App")
st.text("Build with Streamlit and OpenCV")

if "photo" not in st.session_state:
	st.session_state["photo"]="not done"

c2, c3 = st.columns([2,1])
def change_photo_state():
	st.session_state["photo"]="done"

face_cascade = cv2.CascadeClassifier('frecog/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('frecog/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('frecog/haarcascade_smile.xml')

def detect_faces(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	# Detect faces
	faces = face_cascade.detectMultiScale(gray, 1.1, 4)
	# Draw rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
	return img,faces 


def detect_eyes(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
	for (ex,ey,ew,eh) in eyes:
	    cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	return img

def detect_smiles(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	# Detect Smiles
	smiles = smile_cascade.detectMultiScale(gray, 1.1, 4)
	# Draw rectangle around the Smiles
	for (x, y, w, h) in smiles:
	    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
	return img

def cartonize_image(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
	# Edges
	gray = cv2.medianBlur(gray, 5)
	edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
	#Color
	color = cv2.bilateralFilter(img, 9, 300, 300)
	#Cartoon
	cartoon = cv2.bitwise_and(color, color, mask=edges)

	return cartoon


def cannize_image(our_image):
	new_img = np.array(our_image.convert('RGB'))
	img = cv2.cvtColor(new_img,1)
	img = cv2.GaussianBlur(img, (11, 11), 0)
	canny = cv2.Canny(img, 100, 150)
	return canny
@st.cache
def load_image(img):
	im = Image.open(img)
	return im

activities = ["Detection","About"]
choice = st.sidebar.selectbox("Select Activty",activities)
uploaded_photo = c2.file_uploader("Upload Image",type=['jpg','png','jpeg'], on_change=change_photo_state)
camera_photo = c2.camera_input("Take a photo", on_change=change_photo_state)

if choice == 'Detection':
	st.subheader("Face Detection") 
	if st.session_state["photo"]=="done":
		if uploaded_photo:
			our_image= load_image(uploaded_photo)
		if camera_photo:
			our_image= load_image(camera_photo)
		if uploaded_photo==None and camera_photo==None:
			our_image=load_image("image.jpg")
		enhance_type = st.sidebar.radio("Enhance Type",["Original","Gray-Scale","Contrast","Brightness","Blurring"])
		if enhance_type == 'Gray-Scale':
			new_img = np.array(our_image.convert('RGB'))
			img = cv2.cvtColor(new_img,1)
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			# st.write(new_img)
			st.image(gray)
		elif enhance_type == 'Contrast':
			c_rate = st.sidebar.slider("Contrast",0.5,3.5)
			enhancer = ImageEnhance.Contrast(our_image)
			img_output = enhancer.enhance(c_rate)
			st.image(img_output)

		elif enhance_type == 'Brightness':
			c_rate = st.sidebar.slider("Brightness",0.5,3.5)
			enhancer = ImageEnhance.Brightness(our_image)
			img_output = enhancer.enhance(c_rate)
			st.image(img_output)

		elif enhance_type == 'Blurring':
			new_img = np.array(our_image.convert('RGB'))
			blur_rate = st.sidebar.slider("Brightness",0.5,3.5)
			img = cv2.cvtColor(new_img,1)
			blur_img = cv2.GaussianBlur(img,(11,11),blur_rate)
			st.image(blur_img)
		elif enhance_type == 'Original':
			
			st.image(our_image,width=300)
		else:
			st.image(our_image,width=300)


	# Face Detection
	task = ["Faces","Smiles","Eyes","Cannize","Cartonize"]
	feature_choice = st.sidebar.selectbox("Find Features",task)
	if st.button("Process"):

		if feature_choice == 'Faces':
			result_img,result_faces = detect_faces(our_image)
			st.image(result_img)

			st.success("Found {} faces".format(len(result_faces)))
		elif feature_choice == 'Smiles':
			result_img = detect_smiles(our_image)
			st.image(result_img)


		elif feature_choice == 'Eyes':
			result_img = detect_eyes(our_image)
			st.image(result_img)

		elif feature_choice == 'Cartonize':
			result_img = cartonize_image(our_image)
			st.image(result_img)

		elif feature_choice == 'Cannize':
			result_canny = cannize_image(our_image)
			st.image(result_canny)
elif choice == 'About':
	st.subheader("About Face Detection App")
	st.markdown("Built with Streamlit by [Soumen Sarker](https://soumenksarker-personal-website-1--homepage-aqokf8.streamlitapp.com/)")
	st.success("Isshor Saves @Soumen Sarker")