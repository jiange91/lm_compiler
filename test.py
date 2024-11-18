import signal
import time

class GracefulExiter():

    def __init__(self):
        self.state = False
        signal.signal(signal.SIGINT, self.change_state)

    def change_state(self, signum, frame):
        print("exit flag set to True (repeat to exit now)")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self.state = True

    def exit(self):
        return self.state


x = 1
flag = GracefulExiter()
while True:
    print("Processing file #",x,"started...")
    time.sleep(1)
    x+=1
    print(" finished.")
    if flag.exit():
        break