#!/usr/bin/env python3

import whisper
from pyaudio import PyAudio, paInt16
import numpy as np
import torch
import tkinter as tk
from threading import Thread
import subprocess

class SpeechRecognitionApp:
    def __init__(self,root):
        self.root = root
        # PTT button
        self.button = tk.Button(root, text="PTT", bg="orange", highlightbackground='black')
        self.button.bind("<Button-1>", self.start_recognition)
        self.button.bind("<ButtonRelease-1>", self.stop_recognition)
        self.button.grid(row=0, column=0, columnspan=2, sticky="nsew")
        # STT output
        self.label = tk.Label(root, text="", bg="black", fg="white")
        self.label.grid(row=1, column=0, columnspan=2, sticky="nsew")
        # Model output
        self.output_text_frame = tk.Frame(root, bg="black")
        self.output_text_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")

        # Configure grid to make widgets resizable
        root.grid_rowconfigure(0, weight=0)
        root.grid_rowconfigure(1, weight=0)
        root.grid_rowconfigure(2, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)

        # Audio setup
        self.model = whisper.load_model("turbo")
        self.audio = PyAudio()
        self.stream = None
        self.recording = False
        self.audio_data = np.array([], dtype=np.int16)
        

    def start_recognition(self, event):
        self.button.config(text="Speaking...")
        self.stream = self.audio.open(format=paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        self.recording = True
        thread = Thread(target=self.record_audio)
        thread.start()

    def record_audio(self):
        while self.recording:
            data = np.frombuffer(self.stream.read(1024), dtype=np.int16)
            self.audio_data = np.append(self.audio_data, data)
        
        self.stream.stop_stream()
        self.stream.close()
        self.transcribe_audio()

    def transcribe_audio(self):
        audio_data_float = self.audio_data / (2**15 - 1)
        
        result = self.model.transcribe(torch.from_numpy(audio_data_float).float(), language="it")
        text = result["text"]

        display_text = f"Tu: {text}"
        print(display_text)
        self.label.config(text=display_text)

        # ShellGPT
        process = subprocess.Popen(['sgpt', '--shell', '--no-interaction'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        output, _ = process.communicate(input=text.encode())

        if output:
            response = output.decode().strip()
            print("sgpt:", response)
            
            # cancelliamo eventuali righe da risultati precedenti
            for widget in self.output_text_frame.winfo_children():
                widget.destroy()

            lines = response.split('\n')
            for i, line in enumerate(lines):
                frame = tk.Frame(self.output_text_frame, bg='black')
                frame.pack(side=tk.TOP, fill="x")
                if line.strip():  # Ignora le linee vuote
                    button = tk.Button(frame, text="⮞", command=lambda line=line: self.execute_command(line), fg='white', bg='black', highlightbackground='black')
                    button.pack(side=tk.LEFT)
                    entry = tk.Entry(frame, bg="black", fg="white", bd=0, highlightthickness=0)
                    entry.insert(tk.END, f" {line}")
                    entry.pack(fill="x")

        # reimposto il contenuto dell'array
        self.audio_data = np.array([], dtype=np.int16)
        

    def execute_command(self, command):
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Errore '{command}': {e}")


    def stop_recognition(self, event):
        self.button.config(text="PTT")
        self.recording = False


def main():
    root = tk.Tk()
    root.title('JARVIS Linux')
    root.geometry("960x540")
    root.configure(bg='black')
    app = SpeechRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()