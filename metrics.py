# metrics.py
import queue

# single queue instance that both the Flask app and the LiveKit agent import
metrics_q: queue.Queue = queue.Queue()
