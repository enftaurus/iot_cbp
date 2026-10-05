"""
Buzzer module for alarm control.
Wraps hardware control for Pin 11 buzzer.
"""

from hardware import buzzer_on, buzzer_off, set_buzzer

__all__ = ["buzzer_on", "buzzer_off", "set_buzzer"]
