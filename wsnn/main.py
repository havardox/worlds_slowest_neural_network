import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import Animation
from collections import deque
import matplotlib.pyplot as plt
import matplotlib
import random
import pickle
from pathlib import Path
from PIL import Image, ImageTk
import threading
import time
from typing import Sequence

matplotlib.use("TkAgg")

from wsnn.neural_network import NeuralNetwork
from wsnn.data import DataPoint


class MyWindow:
    def __init__(
        self, win: tk.Tk, network: NeuralNetwork, train_data: Sequence[DataPoint]
    ):
        self.win = win
        self.network = network
        self.train_data = train_data

        # <a href="https://www.freepik.com/free-vector/realistic-carbon-fiber-texture-3d-background_17820231.htm#query=carbon%20pattern&position=0&from_view=keyword&track=ais">Image by starline</a> on Freepik
        self.bgimg = Image.open("SL-092619-23740-39.jpg")  # Load the background image
        self.background_lbl = tk.Label(win)
        self.background_lbl.place(
            x=0, y=0, relwidth=1, relheight=1
        )  # Make background_lbl to always fit the parent window
        self.background_lbl.bind(
            "<Configure>", self.on_resize
        )  # on_resize will be executed whenever background_lbl is resized

        tk.Label(win).grid(row=0, column=0)

        # Create left and right frames
        left_frame = tk.Frame(win)
        left_frame.grid(row=0, padx=10, pady=5, column=0)

        right_frame = tk.Frame(win)
        right_frame.grid(row=0, column=1, sticky="nsew")

        tool_bar = tk.Frame(left_frame, width=180, height=185)
        tool_bar.grid(padx=5, pady=5)

        self._training_queue = deque(maxlen=1)
        self._training = threading.Event()
        self._update_label_task = None

        # ---- Buttons -------
        self.train_btn = tk.Button(tool_bar, text="Train")
        self.train_btn.grid(
            row=1, column=0, padx=5, pady=5, sticky="w" + "e" + "n" + "s"
        )
        self.train_btn.bind("<Button-1>", self.train_btn_pressed)

        show_output_btn = tk.Button(tool_bar, text="Show current output")
        show_output_btn.grid(
            row=2, column=0, padx=5, pady=5, sticky="w" + "e" + "n" + "s"
        )
        show_output_btn.bind("<Button-1>", self.plot)

        print_output_btn = tk.Button(tool_bar, text="Print output to console")
        print_output_btn.grid(
            row=3, column=0, padx=5, pady=5, sticky="w" + "e" + "n" + "s"
        )
        print_output_btn.bind("<Button-1>", self.print_outputs)

        # ---- Figure -------
        fig, (self.subplot1, self.subplot2) = plt.subplots(
            2, gridspec_kw={"height_ratios": [3, 1]}
        )
        fig.set_figheight(5)
        fig.set_figwidth(5)
        fig.suptitle("Neural network")

        # ---- Subplot 1 and 2 -------
        self.cmap = matplotlib.colors.ListedColormap(["red", "blue"])
        bounds = [0, 1]
        self.norm = matplotlib.colors.BoundaryNorm(bounds, self.cmap.N)
        self.subplot1.legend()
        self.subplot1.grid(True)
        self.subplot1.set_xlim(0, 100)
        self.subplot1.set_ylim(0, 100)
        self.plots = FigureCanvasTkAgg(fig, right_frame)
        self.plots.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)

        self.label_txt = "Cost: {cost}"
        self.cost_label = tk.Label(
            right_frame, anchor="w", text=self.label_txt.format(cost=1), width=50
        )
        self.cost_label.grid(row=1, padx=5, pady=5, column=0, sticky="w")

    def on_resize(self, event):
        # resize the background image to the size of label
        image = self.bgimg.resize((event.width, event.height), Image.Resampling.LANCZOS)
        # update the image of the label
        self.background_lbl.image = ImageTk.PhotoImage(image)
        self.background_lbl.config(image=self.background_lbl.image)

    def data(self):
        unselected_inputs = [[], [], []]
        selected_inputs = [[], [], []]

        for data_point in self.train_data:
            classify = self.network.classify(data_point.inputs)
            output_list = selected_inputs
            if data_point.expected_outputs[0]:
                output_list = unselected_inputs
            output_list[0].append(data_point.inputs[0])
            output_list[1].append(data_point.inputs[1])
            output_list[2].append(classify)
        cost = self.network.cost_multiple(self.train_data)
        return unselected_inputs, selected_inputs, cost

    def train_btn_pressed(self, event):
        if self._training.is_set():
            self.win.after_cancel(self._update_label_task)
            self._update_label_task = None
            self._training.clear()
            self.train_thread.join()
            self.train_btn.config(text="Start training")
        else:
            self._training.set()
            self.train_btn.config(text="Stop training")

            self.train_thread = threading.Thread( target=self.train)
            self.train_thread.daemon = True
            self.train_thread.start()
            self.update_label()

    def train(self):
        while self._training.is_set():
            self.network.learn(self.train_data, learn_rate=0.6)
            cost = self.network.cost_multiple(self.train_data)
            self.network.save_weights_and_biases(cost)
            self._training_queue.appendleft(cost)

    def update_label(self):
        try:
            cost = self._training_queue[0]
        except IndexError:
            cost = 1
        self.cost_label.config(text=self.label_txt.format(cost=cost))
        self._update_label_task = self.win.after(ms=100, func=self.update_label)

    def print_outputs(self, event):
        for data_point in self.train_data[:10]:
            print(self.network.layers[0].calculate_outputs(data_point.inputs))

    def plot(self, event):
        unselected_inputs, selected_inputs, cost = self.data()
        self.subplot1.clear()
        self.subplot1.scatter(
            unselected_inputs[0],
            unselected_inputs[1],
            c=unselected_inputs[2],
            cmap=self.cmap,
            norm=self.norm,
        )
        self.subplot1.scatter(
            selected_inputs[0],
            selected_inputs[1],
            c=selected_inputs[2],
            cmap=self.cmap,
            norm=self.norm,
            edgecolors="black",
        )
        self.cost_label.config(text=self.label_txt.format(cost=cost))
        self.plots.draw()


def main():
    random.seed(1)
    train_path = Path("train.pkl")
    if train_path.exists():
        with open(train_path, "rb") as f:
            network = pickle.load(f)
    else:
        network = NeuralNetwork(layer_sizes=[2, 9, 9, 9, 2])
        print(network.layers)

        for layer in network.layers:
            layer.initalize_random_weights()
    train_data = [
        DataPoint(
            inputs=[random.uniform(2, 25) for _ in range(2)],
            label=1,
            num_labels=2,
        )
        for _ in range(50)
    ]
    train_data.extend(
        [
            DataPoint(
                inputs=[random.uniform(0, 100) for _ in range(2)],
                label=0,
                num_labels=2,
            )
            for _ in range(300)
        ]
    )

    window = tk.Tk()
    MyWindow(window, network=network, train_data=train_data)
    window.grid_rowconfigure(0, weight=1)
    window.grid_columnconfigure(0, weight=1)
    window.geometry("800x600")
    window.title("My neural network")
    window.maxsize(1200, 900)  # width x height
    window.mainloop()


if __name__ == "__main__":
    main()
