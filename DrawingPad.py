import tkinter as tk
from Brain import Brain


class DrawingPad:
    def __init__(self, tk_app):
        self.tk_app = tk_app
        self.train_mode = False
        self.current_label = tk.StringVar()
        self.status_label = tk.Label(tk_app, text="Mode: Predict", font=("Arial", 12))
        self.status_label.pack()
        self.train_button = tk.Button(tk_app, text="Enable Train Mode", command=self.toggle_train_mode)
        self.train_button.pack()
        self.label_entry = tk.Entry(tk_app, textvariable=self.current_label)
        self.label_entry.pack()
        self.label_entry.insert(0, "label")
        self.prediction_label = tk.Label(tk_app, text="Prediction: ", font=("Arial", 14, "bold"), fg="blue")
        self.prediction_label.pack()
        self.canvas = tk.Canvas(tk_app, width=500, height=500, bg="black")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.bind("<ButtonPress-3>", self.clear_canvas)  # Optional: keep right-click clear
        self.positions = []

    def toggle_train_mode(self):
        self.train_mode = not self.train_mode
        if self.train_mode:
            self.status_label.config(text="Mode: Train")
            self.train_button.config(text="Disable Train Mode")
        else:
            self.status_label.config(text="Mode: Predict")
            self.train_button.config(text="Enable Train Mode")

    def draw(self, event):
        x = event.x
        y = event.y
        self.positions.append([x, y])
        radius = 12
        self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill="white")
    

    def on_mouse_release(self, event):
        # Only process if there are points
        if not self.positions:
            return
        # Log normalized points
        normalized = Brain.normalizePoints(self.positions)
        print("[LOG] Normalized points:", normalized)

        # Log matrix
        resultGenMatrix = Brain.genNormalMatrix(normalized)
        print("[LOG] Matrix:", resultGenMatrix)

        # Log prediction vector
        prediction_vectors = Brain.flattenMatrix(resultGenMatrix)
        print("[LOG] Prediction vector:", prediction_vectors)

        if self.train_mode:
            label = self.current_label.get().strip()
            if label:
                print(f"[LOG] Training model with label: {label}")
                Brain.trainModel(label, prediction_vectors)
                # Save model after training
                import os
                model_path = os.path.join(os.path.dirname(__file__), "model.json")
                Brain.saveModel(model_path)
                print(f"[LOG] Model updated and saved for label: {label}")
                self.prediction_label.config(text="Prediction: (trained '" + label + "')")
            else:
                print("[LOG] No label provided. Skipping training.")
                self.prediction_label.config(text="Prediction: (no label)")
        else:
            prediction = Brain.predictObject(prediction_vectors)
            print("[LOG] Prediction:", prediction)
            self.prediction_label.config(text="Prediction: " + str(prediction))
        # Clear canvas and reset positions after processing
        self.canvas.delete("all")
        self.positions = []

    def clear_canvas(self, event):
        self.canvas.delete("all")
        self.positions = []
