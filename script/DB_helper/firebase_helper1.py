# import required modules
import firebase_admin
from firebase_admin import db, credentials

cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred, {"databaseURL": "https://esp32-home-932a5-default-rtdb.asia-southeast1.firebasedatabase.app/"})

# creating reference to root node
ref = db.reference("/")

# retrieving data from root node
ref.get()

# update operation (update existing value)
# db.reference("/led").update({"bright": 200})

def updateDb(key, value):
  db.reference("/led").update({key: value})