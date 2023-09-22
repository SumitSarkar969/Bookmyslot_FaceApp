import time
import customtkinter as ctk
import cvzone
import face_recognition
import cv2
import pickle
import numpy as np
from PIL import Image, ImageTk

import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

import pymongo
from bson.objectid import ObjectId

import threading

############################################ Global Variables ###########################################

height = 900
width = 1280

width1 = 600
height1 = 900

# Detection mutex for controling the face recognition script
Detection_mutex = True

profile_pic = None

############################################### Connections #############################################

# firebase storage
cred = credentials.Certificate("Cred/bookmyslot-777-firebase-adminsdk-tbfsw-e6a7af68ef.json")
firebase_admin.initialize_app(cred, {
    "storageBucket": "bookmyslot-777.appspot.com"
})

bucket = storage.bucket()

# mongodb database
clint = pymongo.MongoClient("mongodb+srv://dynamicHarsh:UZt3FXCRpIRxxn1k@cluster0.g4y9fre.mongodb.net")
db = clint["bookMySlot"]
collection = db['doctors']

######################################### Face detection constrains #####################################

file = open('ENCODES1.p', 'rb')
Face_encoding_id = pickle.load(file)
file.close()
Face_encoding, face_ids = Face_encoding_id

# Detected Doctor list
detected = [False] * len(face_ids)


################################################ Functions ##############################################

# Function to update height and width of main image
def Stretch_image(event):
    global width, height
    width = event.width
    height = event.height


def profile_resize(event1):
    global width1, height1
    width1 = event1.width
    height1 = event1.height


# Function to download Profile Picture
def fetch_profile(fetch_id):
    global profile_pic
    # Downloading the profile pic of Doctor
    blob = bucket.get_blob(f'image/{fetch_id}')
    array = np.frombuffer(blob.download_as_string(), np.uint8)
    profile_pic = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
    profile_pic = cv2.cvtColor(profile_pic, cv2.COLOR_BGR2RGB)
    profile_pic = Image.fromarray(profile_pic)


# Function to recognise faces
def Face_detect(img1, curr_frame1):
    global Face_encoding, face_ids, Detection_mutex

    encode_curr_frame = face_recognition.face_encodings(img1, curr_frame1)
    for encodeFaces in encode_curr_frame:

        matches = face_recognition.compare_faces(encodeFaces, Face_encoding)
        distance = face_recognition.face_distance(encodeFaces, Face_encoding)
        match_index = np.argmin(distance)
        if matches[match_index] and distance[match_index] < 0.45:

            # checking if doctor is marked present or not
            if not detected[match_index]:
                detected[match_index] = True

                # updating the database according to the presence of the Doctor
                Filter = {"_id": ObjectId(face_ids[match_index])}
                data = {"$set": {"present": True}}
                collection.update_one(Filter, data)

                # calling a thread to download the image to show
                Th2 = threading.Thread(target=fetch_profile, args=(face_ids[match_index],))
                Th2.start()
                Th2.join()

    time.sleep(2)
    Detection_mutex = True


############################################# Main window Format ########################################

# window
root = ctk.CTk()
root.title('Detector')
root.iconbitmap('Res/Rect.ico')
root.geometry('1220x800')
root.minsize(1220, 600)

# Camera
cam = cv2.VideoCapture(0)
cam.set(3, width)
cam.set(4, height)

########################################### Widgets of the Window #######################################

# root grid config
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=5)
root.rowconfigure(0, weight=1)

# frame configs
frame1 = ctk.CTkFrame(root, corner_radius=16, fg_color='transparent')
frame2 = ctk.CTkFrame(root, corner_radius=16, fg_color='transparent')

# frame positioning
frame1.grid(row=0, column=0, padx=(40, 10), pady=(40, 40), sticky='nsew')
frame2.grid(row=0, column=1, padx=(10, 40), pady=(40, 40), sticky='nsew')

# frame1 grid config
frame1.rowconfigure(0, weight=1)
frame1.rowconfigure(1, weight=4)
frame1.columnconfigure(0, weight=1)

# frame1 -> label config
label1 = ctk.CTkLabel(frame1, text='', bg_color='#61677A')
canvas1 = ctk.CTkCanvas(frame1, highlightthickness=0, background='#61677A')

# frame1 -> label positioning
label1.grid(row=0, pady=(0, 10), sticky='nsew')
canvas1.grid(row=1, pady=(10, 0), sticky='nsew')

# frame2 grid config
frame2.rowconfigure(0, weight=1)
frame2.columnconfigure(0, weight=1)

# frame2 -> canvas config for the realtime webcam
canvas2 = ctk.CTkCanvas(frame2, highlightthickness=0)

# frame2 -> canvas positioning
canvas2.grid(row=0, column=0, sticky='nsew')

############################################# Event Listener ##############################################

# event listener for profile pic resizing
canvas1.bind('<Configure>', profile_resize)
# event listener to resize the real time Webcam resizing
canvas2.bind('<Configure>', Stretch_image)


########################################### Closing window Protocol #######################################
def close_win():
    global running
    running = False


running = True
root.protocol("WM_DELETE_WINDOW", close_win)

##########################################################################################################


while running:
    img = cam.read()[1]
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    small_image = cv2.resize(img, (0, 0), None, 0.25, 0.25)

    curr_frame = face_recognition.face_locations(small_image)

    if len(curr_frame) and Detection_mutex:
        Detection_mutex = False
        Th1 = threading.Thread(target=Face_detect, args=(small_image, curr_frame,))
        Th1.start()

    for face_loc in curr_frame:
        y1, x2, y2, x1 = face_loc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        bbox = x1, y1, x2 - x1, y2 - y1
        img = cvzone.cornerRect(img, bbox, rt=0)

    # profile pic rendering
    if profile_pic is not None:
        resized_profile = profile_pic.resize((width1, height1))
        profile_pic_tk = ImageTk.PhotoImage(resized_profile)
        canvas1.create_image(0, 0, image=profile_pic_tk, anchor='nw')

    # live cam rendering
    img_org = Image.fromarray(img)
    resized_img = img_org.resize((width, height))
    img_tk = ImageTk.PhotoImage(resized_img)
    canvas2.create_image(0, 0, image=img_tk, anchor='nw')

    root.update()
