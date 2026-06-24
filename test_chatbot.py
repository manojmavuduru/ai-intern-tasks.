import unittest
from chatbot import RuleBasedChatbot


class TestRuleBasedChatbot(unittest.TestCase):
    def setUp(self):
        self.bot = RuleBasedChatbot()

    def test_greeting(self):
        response = self.bot.get_response("hello there")
        self.assertTrue(len(response) > 0)

    def test_name_extraction(self):
        response = self.bot.get_response("my name is Alex")
        self.assertIn("Alex", response)

    def test_fallback(self):
        response = self.bot.get_response("asdkjqwlejq")
        self.assertTrue(len(response) > 0)

    def test_history_tracking(self):
        self.bot.get_response("hi")
        self.assertEqual(len(self.bot.history), 2)  # user + bot turn


if __name__ == "__main__":
    unittest.main()
