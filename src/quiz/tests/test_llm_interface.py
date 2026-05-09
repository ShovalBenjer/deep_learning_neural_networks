import unittest
import json

from src.quiz.llm_interface import LLMInterface


class TestLLMInterface(unittest.TestCase):
    def test_default_model(self):
        llm = LLMInterface()
        self.assertEqual(llm.model, "z-ai/glm-5.1")

    def test_custom_model(self):
        llm = LLMInterface(model="custom-model")
        self.assertEqual(llm.model, "custom-model")

    def test_api_key_from_env(self):
        import os
        os.environ["OPENROUTER_API_KEY"] = "test-key-123"
        llm = LLMInterface()
        self.assertEqual(llm.api_key, "test-key-123")
        del os.environ["OPENROUTER_API_KEY"]

    def test_api_key_from_param(self):
        llm = LLMInterface(api_key="param-key")
        self.assertEqual(llm.api_key, "param-key")

    def test_build_question_prompt_multiple_choice(self):
        llm = LLMInterface()
        prompt = llm._build_question_prompt("ReLU", "multiple_choice", "medium")
        self.assertIn("ReLU", prompt)
        self.assertIn("multiple_choice", prompt)
        self.assertIn("medium", prompt)

    def test_build_question_prompt_code_completion(self):
        llm = LLMInterface()
        prompt = llm._build_question_prompt("LSTM", "code_completion", "hard")
        self.assertIn("LSTM", prompt)
        self.assertIn("code_completion", prompt)
        self.assertIn("hard", prompt)

    def test_build_question_prompt_concept_explanation(self):
        llm = LLMInterface()
        prompt = llm._build_question_prompt("Attention", "concept_explanation", "easy")
        self.assertIn("Attention", prompt)
        self.assertIn("concept_explanation", prompt)

    def test_build_evaluation_prompt(self):
        llm = LLMInterface()
        prompt = llm._build_evaluation_prompt("What is ReLU?", "Rectified Linear", "max(0, z)")
        self.assertIn("What is ReLU?", prompt)
        self.assertIn("Rectified Linear", prompt)
        self.assertIn("max(0, z)", prompt)

    def test_parse_json_response(self):
        llm = LLMInterface()
        response = '{"is_correct": true, "score": 1.0, "explanation": "Correct!"}'
        result = llm._parse_json_response(response)
        self.assertTrue(result["is_correct"])
        self.assertEqual(result["score"], 1.0)

    def test_parse_json_response_with_markdown(self):
        llm = LLMInterface()
        response = '```json\n{"is_correct": false, "score": 0.5}\n```'
        result = llm._parse_json_response(response)
        self.assertFalse(result["is_correct"])
        self.assertEqual(result["score"], 0.5)

    def test_parse_json_response_with_surrounding_text(self):
        llm = LLMInterface()
        response = 'Here is the result: {"key": "value"} and some more text'
        result = llm._parse_json_response(response)
        self.assertEqual(result["key"], "value")

    def test_parse_json_response_invalid(self):
        llm = LLMInterface()
        response = "not json at all"
        result = llm._parse_json_response(response)
        self.assertTrue(result.get("parse_error", False))
        self.assertIn("raw_response", result)

    def test_build_question_prompt_defaults_type(self):
        llm = LLMInterface()
        prompt = llm._build_question_prompt("Test", "unknown_type", "easy")
        self.assertIn("multiple_choice", prompt)

    def test_build_evaluation_prompt_without_correct_answer(self):
        llm = LLMInterface()
        prompt = llm._build_evaluation_prompt("Question?", "My answer", None)
        self.assertNotIn("Correct Answer", prompt)


if __name__ == "__main__":
    unittest.main()
