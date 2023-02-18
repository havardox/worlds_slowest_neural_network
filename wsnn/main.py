import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.animation import Animation
from matplotlib import style
import matplotlib.pyplot as plt
import matplotlib
import random
import pickle
from pathlib import Path
from PIL import Image, ImageTk

matplotlib.use("TkAgg")

from wsnn.neural_network import NeuralNetwork
from wsnn.data import DataPoint


def main():
    random.seed(1)
    train_path = Path("train.pkl")
    if train_path.exists():
        with open(train_path, "rb") as f:
            network = pickle.load(f)
    else:
        network = NeuralNetwork(layer_sizes=[2, 6, 6, 6, 2])
        print(network.layers)

        for layer in network.layers:
            layer.initalize_random_weights()
    data = [
        DataPoint(
            inputs=[random.uniform(2, 25) for _ in range(2)],
            label=1,
            num_labels=2,
        )
        for _ in range(50)
    ]
    data.extend(
        [
            DataPoint(
                inputs=[random.uniform(0, 100) for _ in range(2)],
                label=0,
                num_labels=2,
            )
            for _ in range(300)
        ]
    )

    class MyWindow:
        def __init__(self, win):
            # <a href="https://www.freepik.com/free-vector/realistic-carbon-fiber-texture-3d-background_17820231.htm#query=carbon%20pattern&position=0&from_view=keyword&track=ais">Image by starline</a> on Freepik
            self.bgimg = Image.open(
                "SL-092619-23740-39.jpg"
            )  # load the background image
            self.background_lbl = tk.Label(win)
            self.background_lbl.place(
                x=0, y=0, relwidth=1, relheight=1
            )  # make label l to fit the parent window always
            self.background_lbl.bind(
                "<Configure>", self.on_resize
            )  # on_resize will be executed whenever label l is resized

            tk.Label(win).grid(row=0, column=0)

            # Create left and right frames
            left_frame = tk.Frame(win)
            left_frame.grid(row=0, padx=10, pady=5, column=0)

            right_frame = tk.Frame(win)
            right_frame.grid(row=0, column=1, sticky="nsew")

            tk.Label(right_frame, width=50).grid(padx=5, pady=5)

            tool_bar = tk.Frame(left_frame, width=180, height=185)
            tool_bar.grid(padx=5, pady=5)

            # For now, when the buttons are clicked, they only call the self.clicked() method. We will add functionality later.
            tk.Button(tool_bar, text="Train", command=self.clicked).grid(
                row=1, column=0, padx=5, pady=5, sticky="w" + "e" + "n" + "s"
            )
            tk.Button(
                tool_bar, text="Show current inference", command=self.clicked
            ).grid(row=2, column=0, padx=5, pady=5, sticky="w" + "e" + "n" + "s")
            tk.Button(
                tool_bar, text="Print output to console", command=self.clicked
            ).grid(row=3, column=0, padx=5, pady=5, sticky="w" + "e" + "n" + "s")

            # ---- Figure -------
            fig, (ax1, ax2) = plt.subplots(2)
            fig.set_figheight(5)
            fig.set_figwidth(5)
            fig.suptitle('Vertically stacked subplots')

            # ---- Subplot 1 -------
            self.cmap = matplotlib.colors.ListedColormap(["red", "blue"])
            bounds = [0, 1]
            self.norm = matplotlib.colors.BoundaryNorm(bounds, self.cmap.N)
            ax1.grid(True)
            ax1.set_xlim(0, 100)
            ax1.set_ylim(0, 100)
            self.plots = FigureCanvasTkAgg(fig, right_frame)
            self.plots.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)

        def clicked(event):
            """if button is clicked, display message"""
            print("Clicked.")

        def on_resize(self, event):
            # resize the background image to the size of label
            image = self.bgimg.resize(
                (event.width, event.height), Image.Resampling.LANCZOS
            )
            # update the image of the label
            self.background_lbl.image = ImageTk.PhotoImage(image)
            self.background_lbl.config(image=self.background_lbl.image)

        def data(self):
            unselected_inputs = [[], [], []]
            selected_inputs = [[], [], []]

            for data_point in data:
                classify = network.classify(data_point.inputs)
                output_list = selected_inputs
                if data_point.expected_outputs[0]:
                    output_list = unselected_inputs
                output_list[0].append(data_point.inputs[0])
                output_list[1].append(data_point.inputs[1])
                output_list[2].append(classify)
            cost = network.cost_multiple(data)
            return unselected_inputs, selected_inputs, cost

        def start_training(self):
            self.btn1["state"] = "disabled"
            network.learn(data, learn_rate=0.6)
            cost = network.cost_multiple(data)
            network.save_weights_and_biases(cost)
            self.lbl0.config(text=self.lbl0_tmpl.format(cost=cost))
            self.win.after(1, self.start_training)

        def print_outputs(self, event):
            for data_point in data[:10]:
                print(network.layers[0].calculate_outputs(data_point.inputs))

        def plot(self, event):
            unselected_inputs, selected_inputs, cost = self.data()
            self.lbl0.config(text=self.lbl0_tmpl.format(cost=cost))
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
            self.plots.draw()

    window = tk.Tk()
    MyWindow(window)
    window.grid_rowconfigure(0, weight=1)
    window.grid_columnconfigure(0, weight=1)
    window.geometry("800x600")
    window.title("My neural network")
    window.maxsize(1200, 900)  # width x height
    window.mainloop()


if __name__ == "__main__":
    main()
