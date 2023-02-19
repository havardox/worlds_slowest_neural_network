import tkinter as tk

from collections import deque
from threading import Thread

from random import randint
from time import sleep

# Starting out (this is the main/gui thread).

root = tk.Tk()

label = tk.Label(root, text="Original text")
label.pack()

# Means of communication, between the gui & update threads:
message_queue = deque()


# Create a thread, that will periodically emit text updates.
def emit_text():  # The task to be called from the thread.
    while True:  # Normally should check some condition here.
        message_queue.append(f"Random number: {randint(0, 100)}")
        sleep(1)  # Simulated delay (of 1 sec) between updates.


# Create a separate thread, for the emitText task:
thread = Thread(target=emit_text)
# Cheap way to avoid blocking @ program exit: run as daemon:
thread.setDaemon(True)
thread.start()  # "thread" starts running independently.

# Moving on (this is still the main/gui thread).


# Periodically check for text updates, in the gui thread.
# Where 'gui thread' is the main thread,
# that is running the gui event-loop.
# Should only access the gui, in the gui thread/event-loop.
def consume_text():
    try:
        label["text"] = message_queue.popleft()
    except IndexError:
        pass  # Ignore, if no text available.
    # Reschedule call to consumeText.
    root.after(ms=1000, func=consume_text)


consume_text()  # Start the consumeText 'loop'.

root.mainloop()  # Enter the gui event-loop.
