# Task 1: Chatbot with Rule-Based Responses

This directory contains my implementation of a simple conversational chatbot that uses predefined rules and pattern matching (regular expressions) to converse with a human.

## 🚀 How to Run

### **1. Command Line Interface (CLI)**
Run the chatbot interactively directly inside your terminal:
```bash
python rule_based_chatbot.py
```

### **2. Jupyter Notebook**
Explore the step-by-step logic, pattern matching code, and sample outputs:
```bash
jupyter notebook chatbot_notebook.ipynb
```

---

## 🛠️ Implementation Details

### **Core Algorithm: Regular Expression Pattern Matching**
The chatbot parses user inputs and checks them against a series of compiled regular expressions sequentially. We utilize the Python standard library's `re` module with the `re.IGNORECASE` flag to ensure flexibility.

### **Supported Conversation Patterns**
- **Greetings**: `"hello"`, `"hi"`, `"greetings"`, `"hey"`
- **Self-Identity**: `"what is your name"`, `"who are you"`
- **Creator Details**: `"who created you"`, `"who made you"`, `"source code"`
- **Humor/Jokes**: `"tell me a joke"`, `"joke"`, `"funny"`
- **General Queries**:
  - User Names: `"my name is [Name]"` -> responds with `"Nice to meet you, [Name]!"`
  - Knowledge Topics: `"do you know about [Topic]"` -> captures the topic and returns a dynamic response.
- **Time/Date Inquiry**: `"what is the time"`, `"date"`, `"clock"`
- **Goodbye/Exit**: `"bye"`, `"goodbye"`, `"exit"`, `"quit"`

### **Fallback Handling**
If the user inputs a statement that doesn't trigger any regex matches, a fallback response is picked at random from a pre-defined set of helpful prompts, preventing the conversation flow from breaking down.
