"""
Quiz Agent with Spaced Repetition for Deep Learning Concepts
Implements SM-2 variant algorithm for optimal retention scheduling
"""

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum


class QuestionType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"


@dataclass
class Question:
    id: str
    section: str
    topic: str
    question: str
    choices: List[str]
    correct_answer: int
    explanation: str
    difficulty: int = 3
    question_type: QuestionType = QuestionType.MULTIPLE_CHOICE


@dataclass
class UserProgress:
    user_id: str
    question_id: str
    ease_factor: float = 2.5
    repetitions: int = 0
    interval_days: int = 0
    next_review: str = field(default_factory=lambda: datetime.now().isoformat())
    last_result: Optional[bool] = None
    consecutive_correct: int = 0


class SM2Scheduler:
    """SM-2 Spaced Repetition Algorithm (Anki-style)"""
    
    MIN_EASE_FACTOR = 1.3
    INITIAL_INTERVAL = 1
    SECOND_INTERVAL = 6
    
    def calculate_interval(self, repetitions: int, ease_factor: float) -> int:
        if repetitions == 0:
            return self.INITIAL_INTERVAL
        elif repetitions == 1:
            return self.SECOND_INTERVAL
        else:
            return round(repetitions * ease_factor)
    
    def calculate_ease_factor(self, current_ef: float, quality: int) -> float:
        new_ef = current_ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        return max(new_ef, self.MIN_EASE_FACTOR)
    
    def quality_from_correct(self, is_correct: bool, response_time_ms: Optional[int] = None) -> int:
        if is_correct:
            if response_time_ms and response_time_ms < 5000:
                return 5
            return 4
        return 2


class QuizAgent:
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.scheduler = SM2Scheduler()
        self.questions: Dict[str, Question] = {}
        self.user_progress: Dict[str, UserProgress] = {}
        self._load_questions()
        self._load_progress()
    
    def _load_questions(self):
        questions_data = [
            Question(
                id="foundations_01", section="Foundations", topic="Neural Networks Basics",
                question="What are the main components of an artificial neuron?",
                choices=["Dendrites, cell body, axon, synapses", "Inputs, weights, bias, activation function", 
                         "Hidden layers, output layer, input layer", "All of the above"],
                correct_answer=1,
                explanation="Artificial neurons receive inputs, compute weighted sums with bias, and apply activation functions.",
                difficulty=1
            ),
            Question(
                id="foundations_02", section="Foundations", topic="Perceptron",
                question="What is the primary limitation of single-layer perceptrons?",
                choices=["They can only solve linearly separable problems", "They require too much training data",
                         "They are not biologically inspired", "They cannot use ReLU activation"],
                correct_answer=0,
                explanation="Single-layer perceptrons can only learn linearly separable decision boundaries, failing on XOR problems.",
                difficulty=2
            ),
            Question(
                id="activation_01", section="Activation Functions", topic="ReLU",
                question="What is the main advantage of ReLU over Sigmoid activation?",
                choices=["Outputs are always positive", "Mitigates vanishing gradient problem",
                         "Computationally expensive", "Outputs zero-centered values"],
                correct_answer=1,
                explanation="ReLU helps mitigate vanishing gradients by having non-zero gradients for positive inputs.",
                difficulty=2
            ),
            Question(
                id="activation_02", section="Activation Functions", topic="Sigmoid vs Tanh",
                question="Which activation function is zero-centered?",
                choices=["Sigmoid", "ReLU", "Tanh", "Binary Threshold"],
                correct_answer=2,
                explanation="Tanh outputs values between -1 and 1, making it zero-centered unlike Sigmoid (0 to 1).",
                difficulty=2
            ),
            Question(
                id="training_01", section="Training", topic="Gradient Descent",
                question="What distinguishes Stochastic Gradient Descent from Batch Gradient Descent?",
                choices=["SGD uses mini-batches", "SGD updates weights per sample, Batch uses entire dataset",
                         "SGD is only for linear models", "Batch is faster but less accurate"],
                correct_answer=1,
                explanation="SGD updates weights after each training example, while Batch uses the entire dataset.",
                difficulty=3
            ),
            Question(
                id="training_02", section="Training", topic="Backpropagation",
                question="What is the purpose of backpropagation?",
                choices=["Initialize weights", "Update weights based on error gradients",
                         "Normalize input data", "Select the best activation function"],
                correct_answer=1,
                explanation="Backpropagation computes gradients of the loss with respect to weights for optimization.",
                difficulty=3
            ),
            Question(
                id="regularization_01", section="Regularization", topic="Dropout",
                question="Dropout regularization works by:",
                choices=["Adding L2 penalty to weights", "Randomly setting neurons to zero during training",
                         "Increasing learning rate", "Reducing network depth"],
                correct_answer=1,
                explanation="Dropout randomly deactivates neurons during training to prevent co-adaptation.",
                difficulty=2
            ),
            Question(
                id="regularization_02", section="Regularization", topic="Overfitting",
                question="Which technique is NOT typically used for overfitting prevention?",
                choices=["Early stopping", "Data augmentation", "Increasing model complexity", "Batch normalization"],
                correct_answer=2,
                explanation="Increasing model complexity typically worsens overfitting, not prevents it.",
                difficulty=2
            ),
            Question(
                id="cnn_01", section="CNN", topic="Convolution",
                question="What is the key benefit of parameter sharing in CNNs?",
                choices=["Faster training", "Reduced number of parameters, translation invariance",
                         "Better gradient flow", "Zero-centered outputs"],
                correct_answer=1,
                explanation="Parameter sharing reduces parameters and helps detect features regardless of position.",
                difficulty=3
            ),
            Question(
                id="cnn_02", section="CNN", topic="Pooling",
                question="What does max pooling primarily achieve?",
                choices=["Increases feature dimensionality", "Provides translation invariance and downsampling",
                         "Normalizes activations", "Adds non-linearity"],
                correct_answer=1,
                explanation="Max pooling reduces dimensionality while providing translation invariance.",
                difficulty=2
            ),
            Question(
                id="rnn_01", section="RNN", topic="Sequence Modeling",
                question="What problem do LSTMs primarily address in vanilla RNNs?",
                choices=["High computational cost", "Vanishing gradients in long sequences",
                         "Lack of parallelization", "Memory overflow"],
                correct_answer=1,
                explanation="LSTMs use gating mechanisms to maintain gradient flow over long sequences.",
                difficulty=3
            ),
            Question(
                id="rnn_02", section="RNN", topic="LSTM Gates",
                question="LSTM stands for:",
                choices=["Linear State Transfer Memory", "Long Short-Term Memory",
                         "Layered Sequence Training Model", "Low-dimensional Sequential Mapping"],
                correct_answer=1,
                explanation="LSTM (Long Short-Term Memory) networks maintain long-term dependencies via cell states.",
                difficulty=1
            ),
            Question(
                id="transformer_01", section="Transformers", topic="Attention",
                question="Self-attention in transformers primarily computes:",
                choices=["Global average pooling", "Relevance between all positions in a sequence",
                         "Layer normalization", "Residual connections"],
                correct_answer=1,
                explanation="Self-attention computes contextual relationships between all tokens in a sequence.",
                difficulty=3
            ),
            Question(
                id="transformer_02", section="Transformers", topic="Multi-head Attention",
                question="What is the purpose of multi-head attention?",
                choices=["Speed up computation", "Capture different types of relationships simultaneously",
                         "Reduce memory usage", "Enable bidirectional context"],
                correct_answer=1,
                explanation="Multiple attention heads capture diverse relationships in different representation subspaces.",
                difficulty=3
            ),
            Question(
                id="evaluation_01", section="Evaluation", topic="Metrics",
                question="When is F1 Score preferred over Accuracy?",
                choices=["Balanced datasets", "Imbalanced datasets", "Regression tasks", "Binary classification only"],
                correct_answer=1,
                explanation="F1 Score balances precision and recall, making it better for imbalanced data.",
                difficulty=2
            ),
            Question(
                id="evaluation_02", section="Evaluation", topic="Confusion Matrix",
                question="True Positive Rate equals:",
                choices=["TP / (TP + TN)", "TP / (TP + FN)", "TN / (TN + FP)", "FP / (FP + TN)"],
                correct_answer=1,
                explanation="True Positive Rate (Sensitivity) = TP / (TP + FN).",
                difficulty=2
            ),
        ]
        
        for q in questions_data:
            self.questions[q.id] = q
    
    def _load_progress(self):
        try:
            with open(f"progress_{self.user_id}.json", "r") as f:
                data = json.load(f)
                for qid, prog in data.items():
                    self.user_progress[qid] = UserProgress(**prog)
        except FileNotFoundError:
            pass
    
    def _save_progress(self):
        data = {qid: asdict(prog) for qid, prog in self.user_progress.items()}
        with open(f"progress_{self.user_id}.json", "w") as f:
            json.dump(data, f, indent=2)
    
    def get_due_questions(self, limit: int = 10) -> List[Question]:
        now = datetime.now()
        due = []
        
        for qid, question in self.questions.items():
            progress = self.user_progress.get(qid)
            if progress is None or datetime.fromisoformat(progress.next_review) <= now:
                due.append(question)
        
        due.sort(key=lambda q: self._get_priority(q.id))
        return due[:limit]
    
    def _get_priority(self, qid: str) -> float:
        progress = self.user_progress.get(qid)
        if progress is None:
            return 100
        days_overdue = (datetime.now() - datetime.fromisoformat(progress.next_review)).days
        return days_overdue + (5 - progress.ease_factor)
    
    def submit_answer(self, question_id: str, selected_answer: int, 
                      response_time_ms: Optional[int] = None) -> Dict:
        question = self.questions[question_id]
        is_correct = selected_answer == question.correct_answer
        
        progress = self.user_progress.get(
            question_id, 
            UserProgress(user_id=self.user_id, question_id=question_id)
        )
        
        quality = self.scheduler.quality_from_correct(is_correct, response_time_ms)
        progress.ease_factor = self.scheduler.calculate_ease_factor(progress.ease_factor, quality)
        
        if is_correct:
            progress.repetitions += 1
            progress.consecutive_correct += 1
            progress.interval_days = self.scheduler.calculate_interval(
                progress.repetitions, progress.ease_factor
            )
        else:
            progress.repetitions = 0
            progress.consecutive_correct = 0
            progress.interval_days = 1
        
        progress.next_review = (datetime.now() + timedelta(days=progress.interval_days)).isoformat()
        progress.last_result = is_correct
        
        self.user_progress[question_id] = progress
        self._save_progress()
        
        return {
            "correct": is_correct,
            "explanation": question.explanation,
            "next_review": progress.next_review,
            "ease_factor": progress.ease_factor,
            "interval_days": progress.interval_days
        }
    
    def get_next_quiz(self, num_questions: int = 5) -> List[Dict]:
        due_questions = self.get_due_questions(num_questions)
        return [{
            "id": q.id,
            "section": q.section,
            "topic": q.topic,
            "question": q.question,
            "choices": q.choices
        } for q in due_questions]
    
    def get_stats(self) -> Dict:
        total = len(self.questions)
        answered = len(self.user_progress)
        mastered = sum(1 for p in self.user_progress.values() if p.repetitions >= 3)
        
        by_section = {}
        for q in self.questions.values():
            section = q.section
            if section not in by_section:
                by_section[section] = {"total": 0, "mastered": 0}
            by_section[section]["total"] += 1
            if q.id in self.user_progress and self.user_progress[q.id].repetitions >= 3:
                by_section[section]["mastered"] += 1
        
        return {
            "total_questions": total,
            "answered_questions": answered,
            "mastered_questions": mastered,
            "completion_rate": round(answered / total * 100, 1) if total > 0 else 0,
            "mastered_rate": round(mastered / total * 100, 1) if total > 0 else 0,
            "by_section": by_section
        }
    
    def export_anki(self, filename: Optional[str] = None) -> str:
        if filename is None:
            filename = f"anki_export_{self.user_id}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            for qid in self.user_progress.keys():
                if qid in self.questions:
                    q = self.questions[qid]
                    f.write(f"{q.question}\t{' '.join(q.choices)}\t{q.correct_answer}\n")
        
        return filename


def main():
    print("=== Deep Learning Quiz Agent with Spaced Repetition ===\n")
    
    user_id = input("Enter your user ID (or press Enter for 'demo'): ").strip() or "demo"
    agent = QuizAgent(user_id)
    
    while True:
        print(f"\n[Stats] {agent.get_stats()}")
        print("\nOptions:")
        print("1. Take quiz")
        print("2. View stats")
        print("3. Export Anki cards")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            questions = agent.get_next_quiz(5)
            if not questions:
                print("No questions due! Come back later.")
                continue
            
            for i, q in enumerate(questions, 1):
                print(f"\n--- Question {i} ---")
                print(f"Section: {q['section']} | Topic: {q['topic']}")
                print(f"\n{q['question']}")
                for j, choice in enumerate(q['choices'], 1):
                    print(f"  {j}. {choice}")
                
                ans = int(input("Your answer (1-4): "))
                result = agent.submit_answer(q['id'], ans - 1)
                print(f"\n{'✓ Correct!' if result['correct'] else '✗ Incorrect'}")
                print(f"Explanation: {result['explanation']}")
        
        elif choice == "2":
            stats = agent.get_stats()
            print(f"\n=== Progress Report ===")
            print(f"Total Questions: {stats['total_questions']}")
            print(f"Answered: {stats['answered_questions']}")
            print(f"Mastered: {stats['mastered_questions']}")
            print(f"Completion: {stats['completion_rate']}%")
            print(f"Mastery Rate: {stats['mastered_rate']}%")
        
        elif choice == "3":
            filename = agent.export_anki()
            print(f"Anki cards exported to {filename}")
        
        elif choice == "4":
            print("Goodbye!")
            break
        
        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()