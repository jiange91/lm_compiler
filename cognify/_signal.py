import signal
import threading
from typing import Optional

class ExitGracefully():
    def __init__(self, msg, verbose):
        self._interrupted = False
        self._lock = threading.Lock()
        self.msg = msg
        self.verbose = verbose
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        with self._lock:
            """Handle SIGINT signal (Ctrl+C)."""
            if not self._interrupted:
                if self.verbose:
                    print("\nInterrupton detected. Exiting gracefully")
                    print("Press Ctrl + C again to force exit.")
                    if self.msg:
                        print(self.msg)
                self._interrupted = True
            else:
                if self.verbose:
                    print("\nForce exiting...")
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                raise KeyboardInterrupt  # Allow immediate termination on second SIGINT
    
    def should_exit(self):
        with self._lock:
            return self._interrupted
    
_exit_gracefully = None

def _init_exit_gracefully(msg=None, verbose=False):
    global _exit_gracefully
    _exit_gracefully = ExitGracefully(msg, verbose)

def _set_exit_msg(msg: str):
    _exit_gracefully.msg = msg
    
def _should_exit():
    return _exit_gracefully.should_exit()