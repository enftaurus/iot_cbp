"""
Registration Module.
Handles registering a new face:
1. Prompts for person's name.
2. Opens camera.
3. Detects face with YuNet alignment landmarks and captures multiple high-quality samples.
4. Generates representative embedding template (SFace 128-D vector).
5. Saves to storage database.
"""

import cv2
import time
import sys
from camera import Camera
from face_recognition_module import (
    detect_faces_detailed,
    get_face_embedding,
    create_representative_embedding
)
import storage


def render_progress_bar(current, total, bar_length=25):
    """Render ASCII progress bar."""
    percent = float(current) / total
    arrow = '█' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write(f"\rCapturing Face Samples... [{arrow}{spaces}] {int(percent * 100)}%")
    sys.stdout.flush()


def register_new_face(num_samples=15):
    """
    Register a new person's face into the system with aligned deep feature vectors.
    """
    print("\n=================================")
    print("       REGISTER NEW FACE")
    print("=================================")
    
    name = input("Enter person's name: ").strip()
    if not name:
        print("❌ Invalid name. Registration cancelled.")
        return False

    print(f"\nPosition '{name}' in front of the camera.")
    print("Slightly tilt your head or change facial expression to capture complete features.")
    print("Press 'q' at any time to cancel.\n")

    embeddings = []
    
    with Camera() as cam:
        captured_count = 0
        last_sample_time = 0
        
        while captured_count < num_samples:
            ret, frame = cam.read()
            if not ret or frame is None:
                print("\n❌ Failed to read frame from camera.")
                break

            detailed_faces = detect_faces_detailed(frame, score_threshold=0.6)
            display_frame = frame.copy()

            if len(detailed_faces) > 0:
                primary_face = detailed_faces[0]
                x, y, w, h = primary_face["box"]
                score = primary_face.get("score", 0.0)

                # Draw bounding box & target overlay
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Registering: {name} (Quality: {int(score*100)}%)",
                            (x, max(25, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                
                # Capture samples at 0.15s interval for sample variance
                now = time.time()
                if now - last_sample_time >= 0.15:
                    emb = get_face_embedding(frame, primary_face)
                    if emb is not None:
                        embeddings.append(emb)
                        captured_count += 1
                        last_sample_time = now
                        render_progress_bar(captured_count, num_samples)

            else:
                cv2.putText(display_frame, "No face detected. Look directly at camera.", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Draw camera UI overlay
            cv2.putText(display_frame, f"Progress: {captured_count}/{num_samples} samples",
                        (30, display_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            try:
                cv2.imshow("Register New Face - Smart Door Security", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠️ Registration cancelled by user.")
                    cv2.destroyAllWindows()
                    return False
            except Exception:
                pass

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    print() # New line after progress bar

    if len(embeddings) == 0:
        print("❌ Could not capture valid face samples. Registration failed.")
        return False

    print("\nGenerating representative deep neural embedding...")
    rep_embedding = create_representative_embedding(embeddings)
    
    if rep_embedding is None:
        print("❌ Failed to generate representative face embedding.")
        return False

    # Save to storage
    success = storage.save_face(name, rep_embedding)
    
    if success:
        print("\n=================================")
        print("  ✅ Face Registered Successfully!")
        print(f"  Name: {name}")
        print(f"  Captured Samples: {len(embeddings)}")
        print(f"  Embedding Dimensions: {rep_embedding.shape[0]}-D Normalized Vector")
        print("=================================\n")
        return True
    else:
        print("❌ Error saving face to database.")
        return False
