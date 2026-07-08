import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate("dt-rasppi-firebase-adminsdk-fbsvc-2c475bf795.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

db.collection("digital-twin").add({
    "temperature": 8.0,
    "humidity": 73.0,
    "timestamp": firestore.SERVER_TIMESTAMP
})

print("Berhasil mengirim data!")

docs = db.collection("digital-twin").stream()

for doc in docs:
    print(doc.id)
    print(doc.to_dict())