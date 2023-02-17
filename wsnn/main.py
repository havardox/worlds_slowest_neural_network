from tkinter import Tk, Label, Button, RIGHT, BOTH
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
import random
import pickle
from pathlib import Path

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
            self.win = win
            x0, y0 = 40, 50
            self.lbl0_tmpl = "Cost: {cost}"
            self.lbl0 = Label(win, text=self.lbl0_tmpl.format(cost=15))
            self.lbl0.config(font=("Arial", 14))
            self.lbl0.place(x=x0, y=y0)

            # ---- Train button -------
            self.btn1 = Button(win, text="Train!", width=15, height=1)
            self.btn1.pack(side="left", anchor="e", expand=True)
            self.btn1.bind("<Button-1>", lambda event: self.start_training())

            # ---- Show plot button -------
            self.btn2 = Button(win, text="Plot current data", width=15, height=1)
            self.btn2.pack(side="left", anchor="w", expand=True)
            self.btn2.bind("<Button-1>", self.plot)

            # ---- Print outputs button -------
            self.btn3 = Button(win, text="Print outputs", width=15, height=1)
            self.btn3.pack(side="left", anchor="s", expand=True)
            self.btn3.bind("<Button-1>", self.print_outputs)

            # ---- Figure -------
            self.figure = Figure(figsize=(4.5, 3), dpi=100)

            # ---- Subplot 1 -------
            self.subplot1 = self.figure.add_subplot(111)
            self.cmap = matplotlib.colors.ListedColormap(["red", "blue"])
            bounds = [0, 1]
            self.norm = matplotlib.colors.BoundaryNorm(bounds, self.cmap.N)
            self.subplot1.grid(True)
            self.subplot1.set_xlim(0, 100)
            self.subplot1.set_ylim(0, 100)

            # ---- Show the plot-------
            self.plots = FigureCanvasTkAgg(self.figure, win)
            self.plots.get_tk_widget().pack(side=RIGHT, fill=BOTH, expand=0)

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

    window = Tk()
    MyWindow(window)
    window.title("My neural network")
    window.geometry("800x600+10+10")
    window.mainloop()


if __name__ == "__main__":
    main()
