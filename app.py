import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Auto-initialize database on startup (for Render deployment)
try:
    from database.connection import engine
    from database.models import Base
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
        conn.commit()
    Base.metadata.create_all(engine)
    print("Database initialized successfully!")
except Exception as e:
    print(f"Error initializing database: {e}")

from ui.gradio_app import create_ui

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
