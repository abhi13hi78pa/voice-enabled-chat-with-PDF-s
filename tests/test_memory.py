import pytest
from rag.memory import ConversationMemory

def test_memory_initialization():
    mem = ConversationMemory()
    assert mem.max_turns == 5
    assert mem.history == []

def test_add_turn():
    mem = ConversationMemory()
    mem.add_turn("hi", "hello")
    assert len(mem) == 1
    assert mem.history[0] == ("hi", "hello")

def test_max_turns_trimming():
    mem = ConversationMemory(max_turns=5)
    for i in range(7):
        mem.add_turn(f"u{i}", f"a{i}")
    assert len(mem) == 5
    assert mem.history[0] == ("u2", "a2")
    assert mem.history[-1] == ("u6", "a6")

def test_get_context_string_empty():
    mem = ConversationMemory()
    assert mem.get_context_string() == "No previous conversation."

def test_get_context_string_with_history():
    mem = ConversationMemory()
    mem.add_turn("hello", "hi there")
    mem.add_turn("how are you?", "I am fine")
    expected = "User: hello\nAssistant: hi there\nUser: how are you?\nAssistant: I am fine"
    assert mem.get_context_string() == expected

def test_get_last_n_turns():
    mem = ConversationMemory(max_turns=10)
    for i in range(5):
        mem.add_turn(f"u{i}", f"a{i}")
    last_two = mem.get_last_n_turns(2)
    assert len(last_two) == 2
    assert last_two == [("u3", "a3"), ("u4", "a4")]

def test_clear():
    mem = ConversationMemory()
    mem.add_turn("1", "2")
    mem.clear()
    assert mem.history == []
    assert len(mem) == 0

def test_is_empty():
    mem = ConversationMemory()
    assert mem.is_empty() is True
    mem.add_turn("1", "2")
    assert mem.is_empty() is False

def test_len():
    mem = ConversationMemory()
    assert len(mem) == 0
    mem.add_turn("1", "2")
    assert len(mem) == 1

def test_custom_max_turns():
    mem = ConversationMemory(max_turns=2)
    for i in range(3):
        mem.add_turn(f"u{i}", f"a{i}")
    assert len(mem) == 2
    assert mem.history[0] == ("u1", "a1")
    assert mem.history[1] == ("u2", "a2")
