"""Phase 6: Conversation Memory for RAG pipeline."""

class ConversationMemory:
    """Manages multi-turn conversation history for the RAG pipeline."""
    
    def __init__(self, max_turns: int = 5):
        """Initialize with a maximum number of conversation turns to retain."""
        self.max_turns = max_turns
        self.history: list[tuple[str, str]] = []  # (user_message, assistant_response)
    
    def add_turn(self, user_message: str, assistant_response: str) -> None:
        """Add a conversation turn and trim to max_turns."""
        self.history.append((user_message, assistant_response))
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]
    
    def get_context_string(self) -> str:
        """Format conversation history as a string for the LLM prompt."""
        if not self.history:
            return "No previous conversation."
        lines = []
        for user_msg, assistant_msg in self.history:
            lines.append(f"User: {user_msg}")
            lines.append(f"Assistant: {assistant_msg}")
        return "\n".join(lines)
    
    def get_last_n_turns(self, n: int = 3) -> list[tuple[str, str]]:
        """Return the last n turns."""
        return self.history[-n:]
    
    def clear(self) -> None:
        """Clear all conversation history."""
        self.history = []
    
    def is_empty(self) -> bool:
        """Check if there is any conversation history."""
        return len(self.history) == 0
    
    def __len__(self) -> int:
        return len(self.history)
