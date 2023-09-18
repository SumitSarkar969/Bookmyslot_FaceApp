import cv2
import face_recognition
import pickle
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

cred = credentials.Certificate("Cred/bookmyslot-777-firebase-adminsdk-tbfsw-e6a7af68ef.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'bookmyslot-777.appspot.com'
})
bucket = storage.bucket()
sf = bucket.list_blobs()
List = list(sf)
face_ids = []
for i in List:
    face_ids.append((str(i).split(', ')[1]).split('/')[1])
face_list = []
for i in face_ids:
    blob = bucket.get_blob(f'image/{i}')
    array = np.frombuffer(blob.download_as_string(), np.uint8)
    face_list.append(cv2.imdecode(array, cv2.COLOR_BGRA2BGR))


def fencode(faceList):
    Face_encodes = []
    for faces in faceList:
        faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)
        Face_encodes.append(face_recognition.face_encodings(faces)[0])
    return Face_encodes


print('Load start')
Face_encoding = fencode(face_list)
Face_encoding_id = [Face_encoding, face_ids]
print('Done')
file = open("ENCODES1.p", 'wb')
pickle.dump(Face_encoding_id, file)
file.close()
