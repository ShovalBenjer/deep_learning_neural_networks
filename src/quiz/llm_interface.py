import json
import os
import urllib.request
import urllib.error
import logging

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "z-ai/glm-5.1"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = 60


class LLMInterface:
    def __init__(
        self,
        api_key=None,
        model=DEFAULT_MODEL,
        base_url=DEFAULT_BASE_URL,
        timeout=DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def _make_request(self, messages, temperature=0.7, max_tokens=1024):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/deep-learning-quiz",
            "X-Title": "Deep Learning Quiz Agent",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url, data=data, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            logger.error("LLM HTTP %s: %s", e.code, error_body)
            raise RuntimeError(f"LLM request failed (HTTP {e.code}): {error_body}") from e
        except urllib.error.URLError as e:
            logger.error("LLM URL error: %s", e.reason)
            raise RuntimeError(f"LLM request failed (URL error): {e.reason}") from e
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error("LLM response parse error: %s", e)
            raise RuntimeError(f"Failed to parse LLM response: {e}") from e

    def generate_quiz_question(self, topic, question_type, difficulty="medium"):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert deep learning educator creating quiz questions. "
                    "Return ONLY valid JSON with no additional text. "
                    "The JSON must follow the exact schema for the requested question type."
                ),
            },
            {
                "role": "user",
                "content": self._build_question_prompt(topic, question_type, difficulty),
            },
        ]
        response = self._make_request(messages, temperature=0.8, max_tokens=800)
        return self._parse_json_response(response)

    def evaluate_answer(self, question, user_answer, correct_answer=None):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert deep learning educator evaluating student answers. "
                    "Return ONLY valid JSON with keys: "
                    '"is_correct" (bool), "score" (0.0-1.0), "explanation" (str), '
                    '"improvement_suggestion" (str). '
                    "Be generous with partial understanding but strict on core concepts."
                ),
            },
            {
                "role": "user",
                "content": self._build_evaluation_prompt(question, user_answer, correct_answer),
            },
        ]
        response = self._make_request(messages, temperature=0.3, max_tokens=600)
        return self._parse_json_response(response)

    def generate_explanation(self, topic, concept):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert deep learning educator. Provide a clear, "
                    "detailed explanation suitable for a student. "
                    "Return ONLY valid JSON with keys: "
                    '"explanation" (str), "key_points" (list of str), '
                    '"related_topics" (list of str).'
                ),
            },
            {
                "role": "user",
                "content": f"Explain the concept '{concept}' in the context of deep learning topic '{topic}'.",
            },
        ]
        response = self._make_request(messages, temperature=0.5, max_tokens=800)
        return self._parse_json_response(response)

    def suggest_study_path(self, weak_topics, strong_topics):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert deep learning tutor. Based on the student's "
                    "performance, suggest a personalized study path. "
                    "Return ONLY valid JSON with keys: "
                    '"study_path" (list of objects with keys "topic" and "reason"), '
                    '"priority_order" (list of str), "estimated_study_time" (str).'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Weak topics (need review): {json.dumps(weak_topics)}\n"
                    f"Strong topics (performing well): {json.dumps(strong_topics)}\n"
                    "Suggest a study path focusing on weak areas while reinforcing "
                    "connections to strong areas."
                ),
            },
        ]
        response = self._make_request(messages, temperature=0.6, max_tokens=800)
        return self._parse_json_response(response)

    def _build_question_prompt(self, topic, question_type, difficulty):
        base = f"Generate a {difficulty} difficulty quiz question about '{topic}' in deep learning.\n\n"
        if question_type == "multiple_choice":
            base += (
                'Return JSON: {"type": "multiple_choice", "question": "...", '
                '"options": {"A": "...", "B": "...", "C": "...", "D": "..."}, '
                '"correct_answer": "A", "explanation": "..."}'
            )
        elif question_type == "code_completion":
            base += (
                'Return JSON: {"type": "code_completion", "question": "...", '
                '"code_template": "...", "blank_description": "...", '
                '"correct_answer": "...", "explanation": "..."}'
            )
        elif question_type == "concept_explanation":
            base += (
                'Return JSON: {"type": "concept_explanation", "question": "...", '
                '"key_concepts": ["..."], "sample_answer": "...", '
                '"grading_rubric": ["..."]}'
            )
        else:
            base += (
                'Return JSON: {"type": "multiple_choice", "question": "...", '
                '"options": {"A": "...", "B": "...", "C": "...", "D": "..."}, '
                '"correct_answer": "A", "explanation": "..."}'
            )
        return base

    def _build_evaluation_prompt(self, question, user_answer, correct_answer):
        prompt = f"Question: {question}\nStudent's Answer: {user_answer}\n"
        if correct_answer:
            prompt += f"Correct Answer: {correct_answer}\n"
        prompt += "Evaluate the student's answer. Return JSON with is_correct, score (0.0-1.0), explanation, and improvement_suggestion."
        return prompt

    def _parse_json_response(self, response_text):
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            return {"raw_response": text, "parse_error": True}
