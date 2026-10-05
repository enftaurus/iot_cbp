# Raspberry Pi Smart Facial Recognition Door Security System

A lightweight, local, real-time facial recognition access control prototype designed for Raspberry Pi and Enthu Technologies boards.

## Features
- **Local Face Registration**: Captures 15 facial samples per user and generates a representative 128-dimensional embedding vector.
- **Local Persistence**: Stores embeddings in `data/faces.pkl` using Python's `pickle`.
- **Real-Time Facial Recognition**: Live continuous monitoring with OpenCV face detection and distance metric matching.
- **MCP230xx GPIO Interface**: Controls Adafruit MCP230xx I2C expander chip:
  - **Pin 11**: Buzzer Output (Triggered on Access Denied for 2 seconds)
  - **Pins 1, 4, 7**: Green LED Outputs (Active on Access Granted)
- **Laptop & Hardware Fallback (Mock Mode)**: Automatically detects if physical I2C / camera hardware is present. Runs in Mock Mode seamlessly on laptops or headless systems.

---

## Directory Structure
```
facial-door-security/
├── main.py                    # Menu controller & main entrypoint
├── camera.py                  # OpenCV camera capture with mock webcam fallback
├── face_recognition_module.py # Face detection, embedding generation & comparison
├── registration.py            # User registration workflow & sample capture
├── security.py                # Lock / continuous security monitoring mode
├── hardware.py                # MCP230xx I2C GPIO driver (Pins 1, 4, 7 & 11)
├── buzzer.py                  # Buzzer trigger wrapper
├── storage.py                 # Local pickle database manager
├── requirements.txt           # Python dependencies
└── data/                      # Local storage directory
    └── faces.pkl              # Saved embeddings database
```

---

## MCP230xx Pin Setup (Enthu Technologies / Raspberry Pi)
| Pin | Function | Active State | Behavior |
|-----|----------|--------------|----------|
| **Pin 11** | Buzzer Alarm | HIGH (`True`) | ON for 2s when Unknown face detected |
| **Pin 1** | Green LED 1 | HIGH (`True`) | ON when Registered face matched |
| **Pin 4** | Green LED 2 | HIGH (`True`) | ON when Registered face matched |
| **Pin 7** | Green LED 3 | HIGH (`True`) | ON when Registered face matched |

### Hardware Configuration Code in `main.py`
On the Raspberry Pi, ensure I2C is enabled (`sudo raspi-config`) and uncomment/configure the hardware initialization in `main.py`:

```python
import board
import busio
import hardware

# Initialize I2C Bus & MCP23017 at address 0x20
i2c = busio.I2C(board.SCL, board.SDA)
hardware.init_hardware(i2c_bus=i2c, mcp_type="MCP23017", address=0x20)
```

---

## Running the Application

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Application
```bash
python main.py
```

### 3. Usage Options
1. **Register New Face**: Enter person's name, position face in front of camera to capture samples.
2. **Lock / Start Security**: Enters live monitoring mode. Access Granted turns ON Green LEDs (1, 4, 7); Access Denied triggers Buzzer (11). Press `q` to return.
3. **List Registered Users**: Views stored faces.
4. **Delete Registered User**: Remove a user from local storage.
5. **Exit**: Shuts down security system and resets GPIO pins.
