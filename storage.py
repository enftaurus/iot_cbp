"""
Storage module for saving and loading face embeddings locally.
File path default: data/faces.pkl
"""

import os
import pickle

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_FACES_FILE = os.path.join(DEFAULT_DATA_DIR, "faces.pkl")


def ensure_data_dir(data_dir=DEFAULT_DATA_DIR):
    """Ensure that the data directory exists."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)


def load_faces(filepath=DEFAULT_FACES_FILE):
    """
    Load registered face embeddings from local pickle file.
    Returns dictionary: {"PersonName": embedding_array, ...}
    """
    ensure_data_dir(os.path.dirname(filepath))
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                return data
            return {}
    except Exception as e:
        print(f"⚠️ Warning: Could not load faces database ({e}). Starting fresh.")
        return {}


def save_faces(faces_dict, filepath=DEFAULT_FACES_FILE):
    """
    Save face embeddings dictionary to local pickle file.
    """
    ensure_data_dir(os.path.dirname(filepath))
    try:
        with open(filepath, "wb") as f:
            pickle.dump(faces_dict, f)
        return True
    except Exception as e:
        print(f"❌ Error saving faces database: {e}")
        return False


def save_face(name, embedding, filepath=DEFAULT_FACES_FILE):
    """
    Save or update a single face embedding in the faces database.
    """
    faces = load_faces(filepath)
    faces[name] = embedding
    return save_faces(faces, filepath)


def delete_face(name, filepath=DEFAULT_FACES_FILE):
    """
    Delete a person's embedding from the faces database.
    """
    faces = load_faces(filepath)
    if name in faces:
        del faces[name]
        save_faces(faces, filepath)
        return True
    return False


def list_faces(filepath=DEFAULT_FACES_FILE):
    """
    Get a list of registered person names.
    """
    faces = load_faces(filepath)
    return list(faces.keys())
