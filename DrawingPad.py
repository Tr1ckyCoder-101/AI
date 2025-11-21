import tkinter as tk
from Brain import Brain

class DrawingPad:
    def __init__(self, tk_app):
        self.canvas = tk.Canvas(tk_app, width=500, height=500, bg="black")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonPress-3>", self.clear_canvas)
        self.positions = []

    def draw(self, event):
        x = event.x
        y = event.y
        self.positions.append([x, y])
        radius = 12
        self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill="white")
    
    def clear_canvas(self, event):
        self.canvas.delete("all")

        Brain.normalizePoints(self.positions)
        resultGenMatrix = Brain.genNormalMatrix(self.positions)
        prediction_vectors = Brain.flattenMatrix(resultGenMatrix)
        Brain.visualizeMatrix(resultGenMatrix)
        Brain.predictObject(prediction_vectors)
