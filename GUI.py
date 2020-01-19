import tkinter as tk


class GUI:

    # gui constructor
    def __init__(self, master):
        self.window = master
        self.color_frame = tk.Frame(master=self.window, bg='white')
        # self.vector_processor = VectorProccessor() #needs inputs
        self.color_frame.pack()

        # self.window.after(100, self.update_color(self.vector_processor.get_score()))

        self.start_button = tk.Button(master=self.window, label="Start", command=self.start_function)

    # update the color
    def update_colour(self, score):  # needs thresholds
        if score < 0.4:
            self.update_frame('red')
        elif score > 0.4 and score < 0.8:
            self.update_frame('orange')
        else:
            self.update_frame('green')

    def update__frame(self, color):
        self.color_frame = tk.Frame(bg=color)
        self.color_frame.pack()

    def start_function(self):
        print("Recording")


def main():
    root = tk.Tk()
    gui = GUI(root)
    root.mainloop()


main()

if __name__ == "__main__":
    main()
