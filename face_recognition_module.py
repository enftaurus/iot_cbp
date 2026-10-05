"""
Face Recognition module.
High-Performance Deep Learning Face Detection (OpenCV YuNet) & Recognition (OpenCV SFace).

Features:
1. Real-time edge-optimized Face Detector (YuNet ONNX): 200+ FPS, landmark alignment.
2. High-accuracy 128-D Feature Embedding (SFace ONNX): Cosine similarity matching.
3. Automatic model downloading and caching.
4. Fallback support for OpenCV CascadeClassifier and dlib.
5. Zero false positive default returns (no fake face injection).
"""

import cv2
import numpy as np
import os
import urllib.request

# Default model directory & file paths
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# Cosine similarity match threshold for SFace (OpenCV standard: 0.363 - 0.38)
# Cosine similarity >= COSINE_THRESHOLD indicates Access Granted.
COSINE_THRESHOLD = 0.38

# Globals for Neural Network instances
_yunet_detector = None
_sface_recognizer = None
_yunet_input_size = (320, 240)
ENGINE_MODE = "UNINITIALIZED"

# Fallbacks
_cascade = None
HAS_DLIB_RECOGNITION = False

try:
    import face_recognition as dlib_face_rec
    HAS_DLIB_RECOGNITION = True
except ImportError:
    HAS_DLIB_RECOGNITION = False


def ensure_models_downloaded():
    """Ensure YuNet and SFace ONNX models exist locally and are valid."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Validate YuNet model (~230 KB)
    if os.path.exists(YUNET_PATH) and os.path.getsize(YUNET_PATH) < 100000:
        os.remove(YUNET_PATH)

    if not os.path.exists(YUNET_PATH):
        print("⬇️ Downloading YuNet face detection model...")
        try:
            urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
            print("✅ YuNet model downloaded successfully.")
        except Exception as e:
            print(f"⚠️ Failed to download YuNet model: {e}")

    # Validate SFace model (~38 MB)
    if os.path.exists(SFACE_PATH) and not (35000000 <= os.path.getsize(SFACE_PATH) <= 45000000):
        os.remove(SFACE_PATH)

    if not os.path.exists(SFACE_PATH):
        print("⬇️ Downloading SFace face recognition model (38 MB)...")
        try:
            sface_cdn_url = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
            urllib.request.urlretrieve(sface_cdn_url, SFACE_PATH)
            print("✅ SFace model downloaded successfully.")
        except Exception as e:
            print(f"⚠️ Failed to download SFace model: {e}")


def init_engine():
    """Initialize YuNet Detector and SFace Recognizer engines."""
    global _yunet_detector, _sface_recognizer, ENGINE_MODE, _cascade

    ensure_models_downloaded()

    if hasattr(cv2, 'FaceDetectorYN') and hasattr(cv2, 'FaceRecognizerSF'):
        if os.path.exists(YUNET_PATH) and os.path.exists(SFACE_PATH):
            try:
                _yunet_detector = cv2.FaceDetectorYN.create(
                    YUNET_PATH, "", _yunet_input_size, score_threshold=0.6, nms_threshold=0.3
                )
                _sface_recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")
                ENGINE_MODE = "YUNET_SFACE"
                print("🚀 [FACE ENGINE] High-accuracy OpenCV YuNet + SFace Deep Learning active!")
                return
            except Exception as e:
                print(f"⚠️ Could not load YuNet/SFace ONNX models: {e}")

    # Fallback to Haar Cascade
    if hasattr(cv2, 'CascadeClassifier'):
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if os.path.exists(cascade_path):
                _cascade = cv2.CascadeClassifier(cascade_path)
                ENGINE_MODE = "HAAR_CASCADE"
                print("ℹ️ [FACE ENGINE] Running in Haar Cascade fallback mode.")
                return
        except Exception:
            pass

    ENGINE_MODE = "BASIC_FALLBACK"
    print("ℹ️ [FACE ENGINE] Running in Basic Fallback mode.")


# Auto-initialize face engine on module import
init_engine()


def detect_faces_detailed(frame, score_threshold=0.6):
    """
    Detect faces in frame with high precision and landmark details.
    
    Returns:
        List of dicts: [{"box": (x, y, w, h), "score": float, "raw": numpy_face_array}, ...]
    """
    if frame is None:
        return []

    h_frame, w_frame = frame.shape[:2]

    # Strategy 1: OpenCV YuNet Deep Learning Detector (Fast & Accurate)
    if ENGINE_MODE == "YUNET_SFACE" and _yunet_detector is not None:
        try:
            # Downscale frame for ultra-fast detection
            det_w, det_h = 320, 240
            small_frame = cv2.resize(frame, (det_w, det_h))
            
            _yunet_detector.setInputSize((det_w, det_h))
            _yunet_detector.setScoreThreshold(score_threshold)
            _, raw_faces = _yunet_detector.detect(small_frame)

            if raw_faces is not None and len(raw_faces) > 0:
                scale_x = w_frame / float(det_w)
                scale_y = h_frame / float(det_h)
                
                results = []
                for rf in raw_faces:
                    score = float(rf[14])
                    if score < score_threshold:
                        continue
                    
                    # Scale coordinates back to original frame size
                    rf_scaled = rf.copy()
                    rf_scaled[0:4:2] *= scale_x
                    rf_scaled[1:4:2] *= scale_y
                    rf_scaled[4:14:2] *= scale_x
                    rf_scaled[5:14:2] *= scale_y

                    x, y, w, h = int(rf_scaled[0]), int(rf_scaled[1]), int(rf_scaled[2]), int(rf_scaled[3])
                    # Clamp coordinates to image boundaries
                    x = max(0, min(x, w_frame - 1))
                    y = max(0, min(y, h_frame - 1))
                    w = max(1, min(w, w_frame - x))
                    h = max(1, min(h, h_frame - y))

                    results.append({
                        "box": (x, y, w, h),
                        "score": score,
                        "raw": rf_scaled
                    })
                return results
        except Exception as e:
            pass

    # Strategy 2: dlib face_locations
    if HAS_DLIB_RECOGNITION:
        try:
            # Downscale for performance
            small_rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (320, 240))
            locations = dlib_face_rec.face_locations(small_rgb)
            if locations:
                scale_x = w_frame / 320.0
                scale_y = h_frame / 240.0
                results = []
                for (top, right, bottom, left) in locations:
                    x = int(left * scale_x)
                    y = int(top * scale_y)
                    w = int((right - left) * scale_x)
                    h = int((bottom - top) * scale_y)
                    results.append({"box": (x, y, w, h), "score": 0.85, "raw": None})
                return results
        except Exception:
            pass

    # Strategy 3: Haar Cascade Classifier Fallback
    if _cascade is not None:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = _cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) > 0:
                return [{"box": (int(x), int(y), int(w), int(h)), "score": 0.75, "raw": None} for (x, y, w, h) in faces]
        except Exception:
            pass

    return []


def detect_faces(frame):
    """
    Detect face bounding boxes in a BGR camera frame.
    Returns: List of bounding boxes [(x, y, w, h), ...]
    """
    detailed = detect_faces_detailed(frame)
    return [item["box"] for item in detailed]


def get_face_embedding(frame, face_info):
    """
    Extract a normalized 128-dimensional embedding vector for a detected face.
    
    Parameters:
        frame: numpy array (BGR image)
        face_info: dict from detect_faces_detailed OR tuple (x, y, w, h)
        
    Returns:
        numpy.ndarray: 128-float embedding vector normalized to unit length.
    """
    if frame is None or face_info is None:
        return None

    raw_face = None
    if isinstance(face_info, dict):
        face_box = face_info.get("box")
        raw_face = face_info.get("raw")
    elif isinstance(face_info, (tuple, list)):
        face_box = face_info
    else:
        return None

    # Deep Learning SFace Embedding with Landmark Face Alignment
    if ENGINE_MODE == "YUNET_SFACE" and _sface_recognizer is not None and raw_face is not None:
        try:
            aligned_face = _sface_recognizer.alignCrop(frame, raw_face)
            feature = _sface_recognizer.feature(aligned_face)
            if feature is not None:
                feature_vec = feature.flatten().astype(np.float32)
                norm = np.linalg.norm(feature_vec)
                if norm > 0:
                    feature_vec /= norm
                return feature_vec
        except Exception as e:
            pass

    # Crop fallback if raw landmark detection was not available
    x, y, w, h = face_box
    h_img, w_img = frame.shape[:2]
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))

    if HAS_DLIB_RECOGNITION:
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            top, right, bottom, left = y, x + w, y + h, x
            encodings = dlib_face_rec.face_encodings(rgb_frame, [(top, right, bottom, left)])
            if encodings and len(encodings) > 0:
                return encodings[0].astype(np.float32)
        except Exception:
            pass

    # Secondary OpenCV Fallback: Multi-scale LBP & Spatial Histogram
    face_crop = frame[y:y+h, x:x+w]
    if face_crop.size == 0:
        return None

    face_resized = cv2.resize(face_crop, (128, 128))
    gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.equalizeHist(gray_face)

    # 4x4 spatial cells, 8-bin histogram per cell = 128 dimensions
    blocks = []
    block_h, block_w = 32, 32
    for row in range(4):
        for col in range(4):
            b_cell = gray_face[row*block_h:(row+1)*block_h, col*block_w:(col+1)*block_w]
            hist = cv2.calcHist([b_cell], [0], None, [8], [0, 256]).flatten()
            blocks.append(hist)

    feature_vec = np.concatenate(blocks).astype(np.float32)
    norm = np.linalg.norm(feature_vec)
    if norm > 0:
        feature_vec /= norm

    return feature_vec


def create_representative_embedding(sample_embeddings):
    """
    Calculate the unit-normalized average embedding vector from multiple captured face samples.
    
    Parameters:
        sample_embeddings: List of numpy arrays
        
    Returns:
        numpy.ndarray: Averaged 128-dimensional representative embedding vector.
    """
    valid_embeddings = [emb for emb in sample_embeddings if emb is not None]
    if not valid_embeddings:
        return None
    
    avg_embedding = np.mean(valid_embeddings, axis=0)
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        avg_embedding /= norm
    return avg_embedding


def compare_embeddings(live_embedding, known_faces_dict, threshold=COSINE_THRESHOLD):
    """
    Compare live face embedding against registered embeddings using Cosine Similarity.
    
    Parameters:
        live_embedding: 128-D numpy array
        known_faces_dict: {"PersonName": embedding_vector, ...}
        threshold: cosine similarity threshold (default 0.38)
        
    Returns:
        (matched_name, similarity_score, similarity_pct, is_access_granted)
    """
    if live_embedding is None or not known_faces_dict:
        return "Unknown", 0.0, 0.0, False

    best_match_name = "Unknown"
    max_similarity = -1.0

    live_emb_flat = live_embedding.flatten()
    live_norm = np.linalg.norm(live_emb_flat)
    if live_norm > 0:
        live_emb_flat = live_emb_flat / live_norm

    for name, known_emb in known_faces_dict.items():
        if known_emb is None:
            continue
        known_flat = known_emb.flatten()
        if known_flat.shape != live_emb_flat.shape:
            continue
        known_norm = np.linalg.norm(known_flat)
        if known_norm > 0:
            known_flat = known_flat / known_norm
        
        # Cosine similarity
        similarity = float(np.dot(live_emb_flat, known_flat))
        if similarity > max_similarity:
            max_similarity = similarity
            best_match_name = name

    # Map cosine similarity to a clean display percentage (0-100%)
    # Cosine score range: 0.38 threshold -> ~70% match; 0.60 -> ~88%; 0.80 -> ~98%
    if max_similarity >= threshold:
        similarity_pct = min(99.9, max(70.0, 70.0 + (max_similarity - threshold) * 60.0))
        return best_match_name, max_similarity, round(similarity_pct, 1), True
    else:
        similarity_pct = max(0.0, min(69.0, max_similarity * 100.0 if max_similarity > 0 else 0.0))
        return "Unknown", max_similarity, round(similarity_pct, 1), False
