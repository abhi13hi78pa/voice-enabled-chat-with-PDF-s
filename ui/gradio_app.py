import gradio as gr
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
from ingestion.pdf_loader import load_pdf
from ingestion.chunker import chunk_documents

# Global state to keep track of uploaded documents in this session
current_session_docs = []

def process_pdfs(files):
    global current_session_docs
    if not files:
        return "No files uploaded."
    
    status_msg = []
    
    for file_obj in files:
        file_path = file_obj.name
        filename = os.path.basename(file_path)
        doc_id = str(uuid.uuid4())
        
        try:
            docs = load_pdf(file_path, doc_id)
            chunks = chunk_documents(docs)
            
            # Phase 3: Vector Indexing & Embeddings
            from rag.vector_store import index_documents
            num_indexed = index_documents(doc_id, filename, chunks)
            
            current_session_docs.append(doc_id)
            if num_indexed > 0:
                status_msg.append(f"✅ {filename}: Parsed, split into {len(chunks)} chunks, and indexed into pgvector.")
            else:
                status_msg.append(f"ℹ️ {filename}: Already indexed.")
            
        except Exception as e:
            status_msg.append(f"❌ {filename}: Error - {str(e)}")
            
    return "\n".join(status_msg)

def chat_interface(message, history, audio_input):
    global current_session_docs
    
    # [PHASE 1 BOUNDARY]
    # STT pipeline is planned for later phases.
    if audio_input and not message:
        message = "[Audio Input Received - STT decoding pending Phase 9]"
            
    if not message:
        return "", history, None
        
    try:
        from rag.rag_service import generate_answer
        answer_text, citations = generate_answer(message, document_ids=current_session_docs)
        
        # Format the final answer with citations for the UI
        final_answer = answer_text
        if citations:
            final_answer += "\n\n**Sources:**\n" + "\n".join([f"- {c}" for c in citations])
            
    except Exception as e:
        final_answer = f"⚠️ Error: {str(e)}"
    
    # [PHASE 1 BOUNDARY]
    # TTS output planned for Phase 10.
    audio_output = None
    
    history.append((message, final_answer))
    
    return "", history, audio_output

def clear_session():
    global current_session_docs
    current_session_docs = []
    return "Session cleared.", [], None

def create_ui():
    with gr.Blocks(title="VOICEPDF - PDF Assistant") as demo:
        gr.Markdown("# VOICEPDF\n### Voice-Enabled Conversational PDF Assistant (Phase 4: LLM Generation)")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Upload PDFs")
                file_input = gr.File(file_count="multiple", type="filepath")
                process_btn = gr.Button("Process / Index")
                status_output = gr.Textbox(label="Status", interactive=False)
                clear_btn = gr.Button("Clear Session Documents")
                
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Chat")
                chatbot = gr.Chatbot(height=400)
                
                with gr.Row():
                    msg_input = gr.Textbox(label="Type your question here...", scale=4)
                    audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Voice Input", scale=1)
                
                submit_btn = gr.Button("Send")
                audio_output = gr.Audio(label="🔊 Answer Audio", autoplay=True, type="filepath")
                
        # Event wiring
        process_btn.click(fn=process_pdfs, inputs=[file_input], outputs=[status_output])
        
        submit_btn.click(
            fn=chat_interface,
            inputs=[msg_input, chatbot, audio_input],
            outputs=[msg_input, chatbot, audio_output]
        )
        msg_input.submit(
            fn=chat_interface,
            inputs=[msg_input, chatbot, audio_input],
            outputs=[msg_input, chatbot, audio_output]
        )
        clear_btn.click(fn=clear_session, inputs=[], outputs=[status_output, chatbot, audio_output])
        
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
