"""
VoicePDF Gradio UI — Voice-Enabled Conversational PDF Assistant.

Phase 1: Gradio shell with upload and chat interface.
Phase 4: Grounded RAG generation with citations.
Phase 5: Voice input (STT) and audio output (TTS).
Phase 6: Multi-turn conversation memory.
Phase 8: Ingestion service integration.
"""
import gradio as gr
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.service import ingest_pdf
from rag.rag_service import generate_answer
from rag.memory import ConversationMemory
from voice.stt_service import transcribe
from voice.tts_service import synthesize, cleanup_temp_files

logger = logging.getLogger(__name__)

# Global session state
current_session_docs: list[str] = []
conversation_memory = ConversationMemory(max_turns=5)


def process_pdfs(files):
    """Orchestrate PDF ingestion for uploaded files."""
    global current_session_docs
    if not files:
        return "⚠️ No files uploaded. Please select one or more PDF files."

    status_msgs = []

    for file_obj in files:
        file_path = file_obj if isinstance(file_obj, str) else file_obj.name
        filename = os.path.basename(file_path)

        result = ingest_pdf(file_path)

        if result["status"] == "success":
            current_session_docs.append(result["document_id"])
            status_msgs.append(f"✅ {filename}: {result['message']}")
        else:
            status_msgs.append(f"❌ {filename}: {result['message']}")

    return "\n".join(status_msgs)


def chat_interface(message, history, audio_input):
    """Handle user chat messages (text or voice) and return grounded answers."""
    global current_session_docs, conversation_memory

    # --- Handle voice input via STT ---
    if audio_input and (not message or not message.strip()):
        try:
            transcribed = transcribe(audio_input)
            if transcribed:
                message = transcribed
        except Exception as e:
            logger.warning(f"STT transcription failed: {e}")
            message = f"[Voice Input Error: {str(e)}]"

    if not message or not message.strip():
        return "", history, None

    # --- Generate RAG answer with conversation memory ---
    answer_text = ""
    final_answer = ""

    try:
        # Get conversation history for multi-turn support
        chat_history_str = conversation_memory.get_context_string()

        answer_text, citations = generate_answer(
            question=message,
            document_ids=current_session_docs,
            chat_history=chat_history_str,
        )

        # Format the final answer with citations for the UI
        final_answer = answer_text
        if citations:
            final_answer += "\n\n**Sources:**\n" + "\n".join([f"- {c}" for c in citations])

        # Store this turn in conversation memory
        conversation_memory.add_turn(message, answer_text)

    except Exception as e:
        logger.error(f"RAG generation error: {e}")
        final_answer = f"⚠️ Error: {str(e)}"

    # --- Generate voice output via TTS ---
    audio_output = None
    if answer_text and not final_answer.startswith("⚠️ Error:"):
        try:
            audio_output = synthesize(answer_text)
        except Exception as e:
            logger.warning(f"TTS synthesis failed: {e}")
            audio_output = None

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": final_answer})

    return "", history, audio_output


def clear_session():
    """Clear all session state: documents, memory, and audio temp files."""
    global current_session_docs, conversation_memory
    current_session_docs = []
    conversation_memory.clear()
    cleanup_temp_files()
    return "🔄 Session cleared. Upload new PDFs to start fresh.", [], None


def create_ui():
    """Build and return the Gradio Blocks interface."""
    with gr.Blocks(
        title="VoicePDF - Voice-Enabled PDF Assistant",
    ) as demo:
        gr.Markdown(
            "# 🎙️ VoicePDF\n"
            "### Voice-Enabled Conversational PDF Assistant\n"
            "_Upload PDFs, ask questions by text or voice, and get grounded answers with citations._"
        )

        with gr.Row():
            # ---- Left Panel: Document Upload ----
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Document Upload")
                file_input = gr.File(
                    file_count="multiple",
                    type="filepath",
                    label="Upload PDF Files",
                    file_types=[".pdf"],
                )
                process_btn = gr.Button("📥 Process & Index", variant="primary")
                status_output = gr.Textbox(
                    label="Processing Status",
                    interactive=False,
                    lines=4,
                )
                clear_btn = gr.Button("🗑️ Clear Session", variant="secondary")

            # ---- Right Panel: Chat Interface ----
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Chat with your PDFs")
                chatbot = gr.Chatbot(
                    height=400,
                    show_label=False,
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Type your question here...",
                        placeholder="Ask anything about your uploaded PDFs...",
                        scale=4,
                        show_label=False,
                    )
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="🎙️ Voice",
                        scale=1,
                    )

                submit_btn = gr.Button("📤 Send", variant="primary")
                audio_output = gr.Audio(
                    label="🔊 Answer Audio",
                    autoplay=True,
                    type="filepath",
                )

        # ---- Event Wiring ----
        process_btn.click(
            fn=process_pdfs,
            inputs=[file_input],
            outputs=[status_output],
        )

        submit_btn.click(
            fn=chat_interface,
            inputs=[msg_input, chatbot, audio_input],
            outputs=[msg_input, chatbot, audio_output],
        )
        msg_input.submit(
            fn=chat_interface,
            inputs=[msg_input, chatbot, audio_input],
            outputs=[msg_input, chatbot, audio_output],
        )
        clear_btn.click(
            fn=clear_session,
            inputs=[],
            outputs=[status_output, chatbot, audio_output],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
