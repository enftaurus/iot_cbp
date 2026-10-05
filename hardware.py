"""
Hardware Controller for Raspberry Pi & Enthu Technologies Board via MCP230xx I2C.

MCP Pins Assignment:
  - Pin 11: Buzzer output
  - Pin 1:  Green LED 1 output
  - Pin 4:  Green LED 2 output
  - Pin 7:  Green LED 3 output

Provides automatic fallback to Mock Mode when running on a laptop or without I2C hardware.
"""

import time
import sys

# MCP230xx Pin definitions
BUZZER_PIN = 11
GREEN_LED_PINS = [1, 4, 7]
ALL_PINS = [1, 4, 7, 11]

# Global MCP pin objects (if initialized)
_mcp_pins = {}
_is_hardware_available = False


def init_hardware(i2c_bus=None, mcp_type="MCP23017", address=0x20):
    """
    Initialize Adafruit MCP230xx chip and configure pins 1, 4, 7, and 11 as OUTPUTs.
    Returns True if hardware initialized, False if running in Mock Mode.
    """
    global _mcp_pins, _is_hardware_available
    _mcp_pins.clear()
    _is_hardware_available = False

    try:
        import board
        import busio
        import digitalio
        from adafruit_mcp230xx.mcp23017 import MCP23017
        from adafruit_mcp230xx.mcp23008 import MCP23008

        # Create I2C bus
        if i2c_bus is None:
            i2c_bus = busio.I2C(board.SCL, board.SDA)

        # Instantiate MCP chip
        if mcp_type.upper() == "MCP23008":
            mcp = MCP23008(i2c_bus, address=address)
        else:
            mcp = MCP23017(i2c_bus, address=address)

        # Configure Pins 1, 4, 7, 11 as OUTPUTs
        for pin_num in ALL_PINS:
            pin = mcp.get_pin(pin_num)
            pin.direction = digitalio.Direction.OUTPUT
            pin.value = False  # Start LOW
            _mcp_pins[pin_num] = pin

        _is_hardware_available = True
        print("✅ [HARDWARE] MCP230xx hardware successfully initialized.")
        print(f"   Configured OUTPUT pins: Green LEDs (Pins 1, 4, 7), Buzzer (Pin {BUZZER_PIN})")
        return True

    except Exception as e:
        _is_hardware_available = False
        print(f"ℹ️  [HARDWARE INFO] Hardware initialization bypassed/failed ({e}).")
        print("   -> Running in MOCK HARDWARE MODE for Laptop / Desktop testing.")
        return False


def set_pin(pin_num, state: bool):
    """Set individual pin state (HIGH/True or LOW/False)."""
    if _is_hardware_available and pin_num in _mcp_pins:
        try:
            _mcp_pins[pin_num].value = state
        except Exception as e:
            print(f"⚠️ Hardware Pin {pin_num} Error: {e}")
    else:
        state_str = "HIGH 🟢" if state else "LOW 🔴"
        # Print only if needed in debug
        pass


def set_green_leds(state: bool):
    """Turn ON (True) or OFF (False) all green LED pins (1, 4, 7)."""
    for pin in GREEN_LED_PINS:
        set_pin(pin, state)
    if not _is_hardware_available:
        status = "ON (Pins 1, 4, 7) 🟢" if state else "OFF (Pins 1, 4, 7) ⚪"
        print(f"   [MOCK HARDWARE] Green LEDs: {status}")


def set_buzzer(state: bool):
    """Turn ON (True) or OFF (False) the buzzer pin (11)."""
    set_pin(BUZZER_PIN, state)
    if not _is_hardware_available:
        status = "ON (Pin 11) 🔊" if state else "OFF (Pin 11) 🔇"
        print(f"   [MOCK HARDWARE] Buzzer: {status}")


def buzzer_on():
    """Turn ON buzzer (Pin 11)."""
    set_buzzer(True)


def buzzer_off():
    """Turn OFF buzzer (Pin 11)."""
    set_buzzer(False)


def grant_access_signal(display_duration=2.0):
    """
    Signal Access Granted:
    - Green LEDs (Pins 1, 4, 7) -> ON
    - Buzzer (Pin 11) -> OFF
    """
    set_buzzer(False)
    set_green_leds(True)


import threading

_buzzer_lock = threading.Lock()
_buzzer_timer = None


def _pulse_buzzer_thread(duration):
    global _buzzer_timer
    time.sleep(duration)
    with _buzzer_lock:
        set_buzzer(False)
        print("   🔇 BUZZER DEACTIVATED")
        _buzzer_timer = None


def deny_access_signal(buzzer_duration=2.0):
    """
    Signal Access Denied (Non-blocking):
    - Green LEDs (Pins 1, 4, 7) -> OFF
    - Buzzer (Pin 11) -> ON in non-blocking background thread for specified duration
    """
    global _buzzer_timer
    set_green_leds(False)
    with _buzzer_lock:
        set_buzzer(True)
        print("   🔊 BUZZER ACTIVATED (Access Denied)")
        if _buzzer_timer is None or not _buzzer_timer.is_alive():
            _buzzer_timer = threading.Thread(target=_pulse_buzzer_thread, args=(buzzer_duration,), daemon=True)
            _buzzer_timer.start()


def reset_all_pins():
    """Turn off all green LEDs and buzzer."""
    set_green_leds(False)
    set_buzzer(False)
