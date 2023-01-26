import os
from deta import Deta
from dotenv import load_dotenv

load_dotenv(".env")

DETA_KEY = os.getenv("DETA_KEY")

deta = Deta(DETA_KEY)

db = deta.Base("users_db")

def insert_user(username, name, password):
    return db.put({"key":username, "name":name, "password":password})

# insert_user('hello', 'hello', 'hello')

def get_user(username):
    return db.get({"key":username})

def fetch_all_users():
    return db.fetch().items

def delete_user(username):
    return db.delete({"key":username})

def update_user(username, updates):
    return db.update(updates, username)
