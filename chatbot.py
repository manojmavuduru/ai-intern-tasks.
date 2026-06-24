"""
Rule-Based Chatbot
===================
A simple chatbot that uses pattern matching (regular expressions) and
if-else logic to understand user intents and generate appropriate
responses. This demonstrates the foundations of NLP / conversation
flow before moving on to ML-based approaches.

Author: <your name>
"""

import re
import random
import json
import os
from datetime import datetime


class RuleBasedChatbot:
    """A chatbot that matches user input against a set of rules
    (regex patterns) and returns a randomly chosen response from the
    matching rule's response list.
    """

    def __init__(self, rules_path: str = None):
        self.name = "RuleBot"
        self.user_name = None
        self.rules = self._load_rules(rules_path)
        self.history = []

    # ------------------------------------------------------------------
    # Rule loading
    # ------------------------------------------------------------------
    def _load_rules(self, rules_path: str):
        """Load rules from a JSON file if provided, otherwise use the
        built-in default rule set. Each rule has a list of regex
        patterns and a list of possible responses.
        """
        if rules_path and os.path.exists(rules_path):
            with open(rules_path, "r", encoding="utf-8") as f:
                raw_rules = json.load(f)
        else:
            raw_rules = self._default_rules()

        compiled = []
        for rule in raw_rules:
            patterns = [re.compile(p, re.IGNORECASE) for p in rule["patterns"]]
            compiled.append({"patterns": patterns, "responses": rule["responses"]})
        return compiled

    @staticmethod
    def _default_rules():
        return [
            {
                "patterns": [r"\bhello\b", r"\bhi\b", r"\bhey\b"],
                "responses": [
                    "Hello! How can I help you today?",
                    "Hi there! What can I do for you?",
                ],
            },
            {
                "patterns": [r"my name is (\w+)", r"i am (\w+)", r"i'm (\w+)"],
                "responses": ["Nice to meet you, {0}!"],
            },
            {
                "patterns": [r"\bhow are you\b"],
                "responses": [
                    "I'm just a bunch of if-else statements, but I'm doing great! How about you?",
                ],
            },
            {
                "patterns": [r"\btime\b"],
                "responses": ["The current time is {time}."],
            },
            {
                "patterns": [r"\bdate\b|\btoday\b"],
                "responses": ["Today's date is {date}."],
            },
            {
                "patterns": [r"\bjoke\b"],
                "responses": [
                    "Why do programmers prefer dark mode? Because light attracts bugs!",
                    "I would tell you a UDP joke, but you might not get it.",
                ],
            },
            {
                "patterns": [r"\bweather\b"],
                "responses": [
                    "I can't check live weather (I'm rule-based!), but it's always sunny in the terminal.",
                ],
            },
            {
                "patterns": [r"\bthank you\b|\bthanks\b"],
                "responses": ["You're welcome!", "Anytime!"],
            },
            {
                "patterns": [r"\bbye\b|\bgoodbye\b|\bexit\b|\bquit\b"],
                "responses": ["Goodbye! Have a great day.", "See you later!"],
            },
            {
                "patterns": [r"\bhelp\b"],
                "responses": [
                    "You can ask me about the time, date, tell you a joke, or just chat. Try saying 'hello'!"
                ],
            },
        ]

    # ------------------------------------------------------------------
    # Core matching logic
    # ------------------------------------------------------------------
    def get_response(self, user_input: str) -> str:
        self.history.append(("user", user_input))

        for rule in self.rules:
            for pattern in rule["patterns"]:
                match = pattern.search(user_input)
                if match:
                    response = random.choice(rule["responses"])
                    response = self._fill_placeholders(response, match)
                    self.history.append(("bot", response))
                    return response

        fallback = self._fallback_response()
        self.history.append(("bot", fallback))
        return fallback

    def _fill_placeholders(self, response: str, match: re.Match) -> str:
        groups = match.groups()
        if groups:
            response = response.format(*groups)
        response = response.replace("{time}", datetime.now().strftime("%H:%M:%S"))
        response = response.replace("{date}", datetime.now().strftime("%Y-%m-%d"))
        return response

    @staticmethod
    def _fallback_response() -> str:
        fallbacks = [
            "I'm not sure I understand. Could you rephrase that?",
            "Sorry, I didn't quite get that. Try asking something else!",
            "Hmm, that's outside what I know how to respond to.",
        ]
        return random.choice(fallbacks)

    # ------------------------------------------------------------------
    # CLI loop
    # ------------------------------------------------------------------
    def chat(self):
        print(f"{self.name}: Hello! Type 'bye' to exit.")
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            response = self.get_response(user_input)
            print(f"{self.name}: {response}")
            if re.search(r"\bbye\b|\bgoodbye\b|\bexit\b|\bquit\b", user_input, re.IGNORECASE):
                break


if __name__ == "__main__":
    bot = RuleBasedChatbot()
    bot.chat()
