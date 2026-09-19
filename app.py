import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.gradio_app import create_ui

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
