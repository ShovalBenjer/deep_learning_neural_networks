from __future__ import annotations

import json
import os
import sys
from typing import Any

try:
    import openai

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


MODEL_NAME = "z-ai/glm-5.1"


class LLMClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        self._client = None
        if HAS_OPENAI and self.api_key:
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )

    @property
    def available(self) -> bool:
        return self._client is not None

    def _call(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        if not self._client:
            return self._mock_response(system_prompt, user_prompt)
        try:
            response = self._client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            if HAS_OPENAI and isinstance(e, openai.APIError):
                print(f"Warning: LLM API error, falling back to mock: {e}", file=sys.stderr)
            else:
                print(f"Warning: Unexpected error calling LLM, falling back to mock: {e}", file=sys.stderr)
            return self._mock_response(system_prompt, user_prompt)

    def _mock_response(self, system_prompt: str, user_prompt: str) -> str:
        if "generate" in system_prompt.lower() or "quiz" in system_prompt.lower():
            return json.dumps({
                "questions": [
                    {
                        "quiz_type": "multiple_choice",
                        "topic": "neural_networks",
                        "section": "Introduction to Artificial Neural Networks",
                        "question_text": "What is the primary function of an activation function in a neural network?",
                        "options": [
                            {"label": "A", "text": "To normalize input data", "is_correct": False},
                            {"label": "B", "text": "To introduce non-linearity into the network", "is_correct": True},
                            {"label": "C", "text": "To reduce the number of parameters", "is_correct": False},
                            {"label": "D", "text": "To initialize weights", "is_correct": False},
                        ],
                        "correct_answer": "B",
                        "explanation": "Activation functions introduce non-linearity, enabling neural networks to learn complex patterns. Without them, the network would be a simple linear model regardless of depth.",
                        "difficulty": 2,
                    }
                ]
            })
        if "evaluate" in system_prompt.lower() or "grade" in system_prompt.lower():
            return json.dumps({
                "is_correct": True,
                "score": 0.85,
                "feedback": "Your explanation covers the key concept well. Consider also mentioning how non-linearity enables universal function approximation.",
            })
        if "study" in system_prompt.lower() or "path" in system_prompt.lower():
            return json.dumps({
                "weak_topics": ["backpropagation", "gradient descent"],
                "suggested_sections": ["Error Backpropagation", "Gradient Descent"],
                "focus_areas": ["chain rule in backpropagation", "learning rate selection"],
                "reasoning": "Focus on foundational optimization concepts before advancing to architecture-specific topics.",
            })
        if "explain" in system_prompt.lower() or "explanation" in system_prompt.lower():
            return "Activation functions introduce non-linearity, enabling neural networks to learn complex patterns. Without them, the network would be a simple linear model regardless of depth."
        return json.dumps({"error": "Unknown request type"})

    def generate_questions(
        self,
        topic: str,
        section: str,
        quiz_type: str = "multiple_choice",
        num_questions: int = 3,
        difficulty: int = 3,
        context: str = "",
    ) -> list[dict[str, Any]]:
        system_prompt = (
            "You are an expert deep learning educator. Generate quiz questions in JSON format. "
            "Return a JSON object with a 'questions' key containing an array of question objects. "
            "Each question must have: quiz_type, topic, section, question_text, options (array of "
            "{label, text, is_correct} for multiple_choice), correct_answer, explanation, difficulty (1-5). "
            "For code_completion, include 'code_template' with a placeholder like __BLANK__. "
            "For concept_explanation, options should be empty and correct_answer should be a model answer."
        )
        user_prompt = (
            f"Generate {num_questions} {quiz_type} question(s) about '{topic}' "
            f"from the section '{section}' at difficulty {difficulty}/5.\n"
        )
        if context:
            user_prompt += f"Context material:\n{context[:2000]}\n"
        user_prompt += "Return valid JSON only."

        raw = self._call(system_prompt, user_prompt, temperature=0.7)
        try:
            data = json.loads(raw)
            return data.get("questions", [])
        except json.JSONDecodeError:
            return []

    def evaluate_answer(
        self,
        question_text: str,
        correct_answer: str,
        user_answer: str,
        quiz_type: str = "multiple_choice",
    ) -> dict[str, Any]:
        if quiz_type == "multiple_choice":
            is_correct = user_answer.strip().upper() == correct_answer.strip().upper()
            return {
                "is_correct": is_correct,
                "score": 1.0 if is_correct else 0.0,
                "feedback": "Correct!" if is_correct else f"Incorrect. The correct answer is: {correct_answer}",
            }

        system_prompt = (
            "You are an expert deep learning educator evaluating a student's answer. "
            "Return JSON with: is_correct (bool), score (0.0-1.0), feedback (detailed explanation). "
            "Be generous with partial credit for conceptually correct but incomplete answers."
        )
        user_prompt = (
            f"Question: {question_text}\n"
            f"Model answer: {correct_answer}\n"
            f"Student answer: {user_answer}\n"
            f"Evaluate the student's answer. Return valid JSON only."
        )

        raw = self._call(system_prompt, user_prompt, temperature=0.3)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "is_correct": False,
                "score": 0.0,
                "feedback": "Could not evaluate answer. Please try again.",
            }

    def suggest_study_path(
        self, topic_performances: dict[str, Any], all_topics: list[str]
    ) -> dict[str, Any]:
        system_prompt = (
            "You are an expert learning path advisor for deep learning. "
            "Given a student's performance data across topics, suggest a study path. "
            "Return JSON with: weak_topics (array), suggested_sections (array), "
            "focus_areas (array), reasoning (string)."
        )
        perf_summary = {k: {"accuracy": v.accuracy, "total": v.total_questions} for k, v in topic_performances.items()}
        user_prompt = (
            f"Student performance: {json.dumps(perf_summary)}\n"
            f"Available topics: {json.dumps(all_topics)}\n"
            f"Suggest a personalized study path. Return valid JSON only."
        )

        raw = self._call(system_prompt, user_prompt, temperature=0.5)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "weak_topics": list(topic_performances.keys())[:3],
                "suggested_sections": [],
                "focus_areas": [],
                "reasoning": "Could not generate detailed study path. Review all topics.",
            }

    def generate_explanation(
        self, topic: str, question_text: str, correct_answer: str
    ) -> str:
        system_prompt = (
            "You are an expert deep learning educator. Provide a detailed, clear explanation "
            "for the given question and answer. Use examples where helpful."
        )
        user_prompt = (
            f"Topic: {topic}\n"
            f"Question: {question_text}\n"
            f"Correct answer: {correct_answer}\n"
            f"Provide a detailed explanation."
        )
        return self._call(system_prompt, user_prompt, temperature=0.5)
