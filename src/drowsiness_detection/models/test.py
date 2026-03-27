import subprocess

def speak(text):
    cmd = f'PowerShell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"'
    subprocess.Popen(cmd, shell=True)

speak("Hello this is a test")
import time
time.sleep(5)