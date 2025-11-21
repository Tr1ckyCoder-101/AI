import tkinter as tk
from DrawingPad import DrawingPad
from Brain import Brain


import os
model_path = os.path.join(os.path.dirname(__file__), "model.json")
print("[DEBUG] Loading model from:", model_path)
Brain.packBrainModel(model_path)
print("[DEBUG] Model after loading:", Brain.model)

tk_app = tk.Tk()                          
canvas = DrawingPad(tk_app)
tk_app.title("Machine Learning Drawing Pad")


tk_app.mainloop()