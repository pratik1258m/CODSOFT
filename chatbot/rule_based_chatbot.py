#!/usr/bin/env python3
"""
Rule-Based Chatbot
CODSOFT AI Internship - Task 1

This script implements a simple rule-based chatbot that processes user input using 
regular expression pattern matching and returns predefined responses.
"""

import re
import random
import datetime


class RuleBasedChatbot:
    def __init__(self):
        # Define conversation rules: mapping of compiled regex patterns to lists of possible responses
        # The rules are evaluated sequentially, so order matters (more specific patterns first).
        self.rules = [
            (
                r"\b(hi|hello|hey|greetings|hola)\b",
                [
                    "Hello! I am your friendly rule-based assistant. How can I help you today?",
                    "Hi there! How's your day going? What can I do for you?",
                    "Hey! Nice to meet you. How can I assist you today?",
                ]
            ),
            (
                r"\b(good)?bye|see\s+ya|farewell|exit|quit\b",
                [
                    "Goodbye! Have a wonderful day ahead!",
                    "Bye! Feel free to chat again whenever you'd like.",
                    "Farewell! Take care and talk to you soon!",
                ]
            ),
            (
                r"what is your name|who are you",
                [
                    "I am Nexus Bot, a rule-based AI built for the CODSOFT AI Internship!",
                    "You can call me Nexus Bot. I'm a helper chatbot running on predefined rules.",
                ]
            ),
            (
                r"how are you|how is it going|how\'s it going|how do you do",
                [
                    "I'm doing great, thank you for asking! I'm ready to answer your questions.",
                    "I'm operating at peak efficiency! How are you doing today?",
                    "Fantastic! Just waiting to help out. How about yourself?",
                ]
            ),
            (
                r"what can you do|help|features|commands",
                [
                    "I can chat with you, tell jokes, give the current time, and answer basic questions! Try asking: 'tell me a joke', 'what is the time', or 'who made you'."
                ]
            ),
            (
                r"who (created|made|built) you|source code",
                [
                    "I was created by Pratik, an AI engineering intern at CODSOFT!",
                    "My logic was built by Pratik using Python and regular expression pattern matching.",
                ]
            ),
            (
                r"tell me a joke|joke|funny",
                [
                    "Why do programmers wear glasses? Because they can't C#!",
                    "There are 10 types of people in the world: those who understand binary, and those who don't.",
                    "Why did the computer go to the doctor? Because it had a virus!",
                    "How many programmers does it take to change a light bulb? None, that's a hardware problem.",
                ]
            ),
            (
                r"time|date|clock",
                [
                    f"The current local time is {datetime.datetime.now().strftime('%I:%M %p')}.",
                    f"My clock says it's {datetime.datetime.now().strftime('%H:%M:%S')} right now.",
                ]
            ),
            (
                r"\b(love|like) you\b",
                [
                    "Aww, thank you! I enjoy chatting with you too.",
                    "That's so kind of you! I'm here to help you as much as I can.",
                ]
            ),
            (
                r"weather|rain|sunny|temp",
                [
                    "I don't have real-time internet access to check the weather, but I hope it's beautiful and sunny where you are!",
                    "I can't check the weather sensor right now, but a perfect indoor temperature for coding is 21°C!",
                ]
            ),
            # Capture-based greeting rule (e.g. "my name is John")
            (
                r"my name is ([a-zA-Z\s]+)",
                [
                    "Nice to meet you, {0}! How can I help you today?",
                    "Hello {0}! That's a wonderful name. What's on your mind today?",
                ]
            ),
            # General capture-based rule (e.g. "do you know about python")
            (
                r"do you know about ([a-zA-Z\s0-9]+)",
                [
                    "Yes, {0} is a fascinating topic! While I only have predefined rules, you can learn a lot about {0} by studying standard documentation or asking a search engine.",
                    "I've heard about {0}! As a rule-based bot, my knowledge is limited, but {0} sounds highly interesting.",
                ]
            ),
        ]
        
        # Compile all regular expressions for performance and case-insensitivity
        self.compiled_rules = [(re.compile(pattern, re.IGNORECASE), responses) for pattern, responses in self.rules]
        
        # Fallback responses when no rule matches
        self.fallbacks = [
            "I'm not sure I understand that. Can you try rephrasing? (Type 'help' to see what I can do!)",
            "Interesting query! However, my rule-based brain doesn't have a match for that. Ask me a joke instead!",
            "I'm a simple rule-based chatbot. Could you ask something else, like 'what is your name' or 'tell me a joke'?",
            "Hmm, that's beyond my current rules. Try typing 'help' to see my capabilities!",
        ]

    def respond(self, user_input):
        """Processes user input, finds the first matching pattern, and returns an appropriate response."""
        # Clean input
        user_input_clean = user_input.strip()
        if not user_input_clean:
            return "Please say something! I'm listening."

        # Search for a matching rule
        for pattern, responses in self.compiled_rules:
            match = pattern.search(user_input_clean)
            if match:
                response = random.choice(responses)
                # If there are regex groups, format the response with the first group
                if match.groups():
                    group_val = match.group(1).strip()
                    return response.format(group_val)
                return response

        # Return a random fallback response if no match is found
        return random.choice(self.fallbacks)


def main():
    print("=" * 60)
    print("         🤖 WELCOME TO NEXUS RULE-BASED CHATBOT 🤖")
    print("             Created for the CODSOFT AI Internship")
    print("=" * 60)
    print("Type 'help' to see what I can do, or 'exit'/'bye' to quit.\n")
    
    chatbot = RuleBasedChatbot()
    
    while True:
        try:
            user_input = input("You: ")
            # Exit condition
            if re.search(r"\b(good)?bye|exit|quit\b", user_input, re.IGNORECASE):
                # Print chatbot farewell before breaking
                print(f"Bot: {chatbot.respond(user_input)}")
                break
                
            response = chatbot.respond(user_input)
            print(f"Bot: {response}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Goodbye! Have a great day!")
            break


if __name__ == "__main__":
    main()
