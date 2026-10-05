"""
Security Module.
Handles continuous real-time facial recognition security mode:
1. Loads registered embeddings.
2. Captures camera frames.
3. Detects & recognizes faces using YuNet + SFace Deep Neural Engine.
4. Displays live video feed with FPS counter, face bounding boxes, and similarity %.
5. Controls hardware (Green LEDs on match, Buzzer on mismatch).
"""

import cv2
import time
import storage
from camera import Camera
from face_recognition_module import (
    detect_faces_detailed,
    get_face_embedding,
    compare_embeddings,
    ENGINE_MODE
)
import hardware


def start_security_mode(cooldown_seconds=3.0):
    """
    Start continuous facial recognition security mode.
    """
    known_faces = storage.load_faces()
    
    if not known_faces:
        print("\n⚠️ No registered faces found in database!")
        print("Please register at least one face (Option 1) before locking security mode.\n")
        input("Press Enter to return to main menu...")
        return

    print("\n=================================")
    print("          DOOR LOCKED")
    print("=================================")
    print(f"Scanning for faces... [Engine: {ENGINE_MODE}]")
    print("Camera Active")
    print("Status: Waiting for person...")
    print("Press 'q' in camera window or Ctrl+C to stop security mode.\n")

    last_trigger_time = 0
    last_status = "Waiting..."

    # FPS calculation tracking
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0

    with Camera() as cam:
        hardware.reset_all_pins()

        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                print("❌ Failed to capture frame.")
                time.sleep(0.1)
                continue

            fps_frame_count += 1
            now = time.time()
            if now - fps_start_time >= 1.0:
                current_fps = fps_frame_count / (now - fps_start_time)
                fps_frame_count = 0
                fps_start_time = now

            detailed_faces = detect_faces_detailed(frame)
            display_frame = frame.copy()

            if len(detailed_faces) > 0:
                # Target primary face (highest confidence score)
                primary_face = detailed_faces[0]
                x, y, w, h = primary_face["box"]
                score = primary_face.get("score", 0.0)

                live_emb = get_face_embedding(frame, primary_face)

                if live_emb is not None:
                    name, dist_sim, sim_pct, is_granted = compare_embeddings(live_emb, known_faces)

                    if is_granted:
                        # ACCESS GRANTED
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Top banner label
                        label = f"Welcome {name} ({sim_pct}%) - GRANTED"
                        cv2.rectangle(display_frame, (x, max(0, y - 30)), (x + w, y), (0, 255, 0), -1)
                        cv2.putText(display_frame, label, (x + 5, max(18, y - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                        if now - last_trigger_time > cooldown_seconds or last_status != f"GRANTED:{name}":
                            print("\n=================================")
                            print("          ACCESS GRANTED")
                            print("=================================")
                            print(f"Welcome, {name}!")
                            print(f"Face Match: ✓ ({sim_pct}%)")
                            print(f"Detector Confidence: {int(score * 100)}%")
                            print("Access: GRANTED")
                            print("Buzzer: OFF")
                            print("Green LEDs: ON (Pins 1, 4, 7)")
                            print("=================================\n")
                            
                            hardware.grant_access_signal()
                            last_trigger_time = now
                            last_status = f"GRANTED:{name}"

                    else:
                        # ACCESS DENIED
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        
                        label = f"UNKNOWN ({sim_pct}%) - DENIED"
                        cv2.rectangle(display_frame, (x, max(0, y - 30)), (x + w, y), (0, 0, 255), -1)
                        cv2.putText(display_frame, label, (x + 5, max(18, y - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                        
                        if now - last_trigger_time > cooldown_seconds or last_status != "DENIED":
                            print("\n=================================")
                            print("          ACCESS DENIED")
                            print("=================================")
                            print("Unknown Person")
                            print(f"Similarity: {sim_pct}% (Below threshold)")
                            print("Access: DENIED")
                            print("⚠ BUZZER ACTIVATED (Pin 11)")
                            print("=================================\n")
                            
                            hardware.deny_access_signal(buzzer_duration=1.5)
                            last_trigger_time = now
                            last_status = "DENIED"

            else:
                # No face visible
                cv2.putText(display_frame, "Status: Waiting for person...", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                
                # Reset hardware pins to default standby after cooldown
                if now - last_trigger_time > cooldown_seconds:
                    hardware.reset_all_pins()
                    last_status = "Waiting..."

            # Security UI status overlay
            h_img, w_img = display_frame.shape[:2]
            cv2.putText(display_frame, f"FPS: {current_fps:.1f} | Engine: {ENGINE_MODE}", (20, h_img - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(display_frame, "SECURITY MODE ACTIVE [Press 'q' to unlock]", (20, h_img - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            try:
                cv2.imshow("Smart Door Security - Live Monitor", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n🔓 Exiting Security Mode...")
                    break
            except Exception:
                pass

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    
    hardware.reset_all_pins()
    print(" Door Security disarmed. Returned to main menu.\n")
