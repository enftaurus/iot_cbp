"""
Smart Facial Recognition Door Security System
Main application entry point.

Raspberry Pi / Enthu Board Pin Mapping (MCP230xx I2C):
  - Pin 11: Buzzer (Active HIGH on Access Denied)
  - Pin 1, 4, 7: Green Light LEDs (Active HIGH on Access Granted)
"""

import os
import sys
import storage
import hardware
from registration import register_new_face
from security import start_security_mode

# ==============================================================================
# HARDWARE INITIALIZATION (RPI / MCP230xx)
# ==============================================================================
# To explicitly configure hardware when deploying to the Raspberry Pi:
#
# import board
# import busio
# from adafruit_mcp230xx.mcp23017 import MCP23017
#
# i2c = busio.I2C(board.SCL, board.SDA)
# hardware.init_hardware(i2c_bus=i2c, mcp_type="MCP23017", address=0x20)
#
# Note: hardware.init_hardware() is auto-called below with fallback to mock mode
# when running on laptops without I2C pins.
# ==============================================================================


def print_header():
    print("\n" + "="*35)
    print("     SMART DOOR SECURITY")
    print("="*35)


def list_registered_users():
    """List all registered face names in local database."""
    faces = storage.list_faces()
    print("\n=================================")
    print("    REGISTERED USERS DATABASE")
    print("=================================")
    if not faces:
        print("  (No registered users found)")
    else:
        for idx, name in enumerate(faces, 1):
            print(f"  {idx}. {name}")
    print("=================================\n")


def delete_user_prompt():
    """Prompt user to delete a face record."""
    faces = storage.list_faces()
    if not faces:
        print("\n⚠️ Database is empty.")
        return
    
    list_registered_users()
    name = input("Enter name to delete (or press Enter to cancel): ").strip()
    if name in faces:
        if storage.delete_face(name):
            print(f"✅ User '{name}' successfully deleted.")
        else:
            print(f"❌ Failed to delete user '{name}'.")
    elif name:
        print(f"❌ User '{name}' not found in database.")


def main():
    # Initialize MCP230xx hardware (or mock mode on laptop)
    hardware.init_hardware()

    while True:
        print_header()
        print("1. Register New Face")
        print("2. Lock / Start Security")
        print("3. List Registered Users")
        print("4. Delete Registered User")
        print("5. Exit")
        print("="*35)
        
        choice = input("\nEnter option: ").strip()

        if choice == '1':
            register_new_face()
        elif choice == '2':
            start_security_mode()
        elif choice == '3':
            list_registered_users()
        elif choice == '4':
            delete_user_prompt()
        elif choice == '5':
            print("\nShutting down Smart Door Security system. Goodbye!\n")
            hardware.reset_all_pins()
            sys.exit(0)
        else:
            print("\n❌ Invalid option. Please select 1, 2, 3, 4, or 5.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSystem interrupted by user. Exiting...")
        hardware.reset_all_pins()
        sys.exit(0)
