"""An ELIZA-style reflective chatbot.

The bot holds a short, empathetic conversation using regex pattern matching
(in the spirit of Weizenbaum's ELIZA). Once the user has shared enough
(>= MIN_WORDS words) and types "good-bye", the full transcript is written to
an output file, which can then be scored by ``classifier.py``.

Example
-------
    python -m src.chatbot --output output.txt
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import List, Optional, Tuple

from nltk.chat.util import reflections

MIN_WORDS = 31
EXIT_COMMAND = "good-bye"

# Each pair is (compiled-pattern, list-of-response-templates).
PAIRS: List[Tuple[str, List[str]]] = [
    (r"i need (.*)", [
        "Why do you need {0}?",
        "Would it really help you to get {0}?",
        "Are you sure you need {0}?",
    ]),
    (r"why can'?t i ([^?]*)\??", [
        "Do you think you should be able to {0}?",
        "If you could {0}, what would you do?",
        "Have you really tried?",
    ]),
    (r"(.*) computer(.*)", [
        "Does it seem strange to talk to a computer?",
        "How do computers make you feel?",
        "Do you feel threatened by computers?",
    ]),
    (r"i can'?t (.*)", [
        "How do you know you can't {0}?",
        "Perhaps you could {0} if you tried.",
        "What would it take for you to {0}?",
    ]),
    (r"i am (.*)", [
        "Did you come to me because you are {0}?",
        "How long have you been {0}?",
        "How do you feel about being {0}?",
    ]),
    (r"i'?m (.*)", [
        "How does being {0} make you feel?",
        "Why do you think you're {0}?",
    ]),
    (r"i feel (.*)", [
        "Good, tell me more about these feelings.",
        "Do you often feel {0}?",
        "When you feel {0}, what do you do?",
    ]),
    (r"i think (.*)", [
        "Do you doubt {0}?",
        "Do you really think so?",
        "But you're not sure {0}?",
    ]),
    (r"because (.*)", [
        "Is that the real reason?",
        "What other reasons come to mind?",
    ]),
    (r"(.*) sorry (.*)", [
        "There are many times when no apology is needed.",
        "What feelings do you have when you apologize?",
    ]),
    (r"hello(.*)", [
        "Hello... I'm glad you could drop by today.",
        "Hi there... how are you feeling today?",
    ]),
    (r"(.*) mother(.*)", [
        "Tell me more about your mother.",
        "How do you feel about your mother?",
        "Did you try to talk to your mother about the problems?",
    ]),
    (r"(.*) father(.*)", [
        "Tell me more about your father.",
        "How did your father make you feel?",
        "Do you think you should talk to your father about the problems you face?",
    ]),
    (r"(.*) child(.*)", [
        "Did you have close friends since childhood?",
        "What is your favorite childhood memory?",
        "How do you think your childhood experiences relate to your feelings today?",
    ]),
    (r"(.*) friend(.*)", [
        "Tell me more about your friends.",
        "When you think of a friend, who comes to mind?",
        "Do you have close friends you can talk to?",
    ]),
    (r"(.*) job(.*)", [
        "Tell me more about your job.",
        "What problems are you facing at your job?",
        "Are you satisfied with the work you are doing?",
    ]),
    (r"(.*) wife(.*)", [
        "Tell me more about your wife.",
        "What problems are you facing with your wife?",
        "Did you try talking to your wife about this?",
    ]),
    (r"(.*)\?", [
        "Why do you ask that?",
        "Don't you think the answer lies within yourself?",
        "Why don't you tell me the answer to that?",
    ]),
    (r"(.*)", [
        "Please tell me more.",
        "Let's change focus a bit. Tell me about your family and friends.",
        "Can you elaborate on that?",
        "How does that make you feel?",
    ]),
]

_COMPILED = [(re.compile(pattern, re.IGNORECASE), responses) for pattern, responses in PAIRS]


def _reflect(fragment: str) -> str:
    """Swap first/second person pronouns (e.g. 'i am' -> 'you are')."""
    return " ".join(reflections.get(word, word) for word in fragment.lower().split())


def respond(message: str) -> str:
    """Return the bot's reply to a single user message."""
    for pattern, responses in _COMPILED:
        match = pattern.match(message.rstrip(".!"))
        if match:
            groups = [_reflect(g) for g in match.groups()]
            return random.choice(responses).format(*groups)
    return "Please tell me more."


def run(output_path: Path) -> None:
    transcript: List[str] = []
    print("Hi! Please answer in full sentences rather than one or two words.")
    print("We're happy to help. :)")
    print(f"When you'd like to finish, just type '{EXIT_COMMAND}'.")
    print("\nHello. How are you feeling today?")

    while True:
        try:
            message = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if message == EXIT_COMMAND:
            if len(" ".join(transcript).split()) >= MIN_WORDS:
                Path(output_path).write_text(" ".join(transcript), encoding="utf-8")
                print(f"\nThank you. Your responses were saved to {output_path}.")
                break
            print("Let's chat a little more so we can help. Tell me more about your problem.")
            continue

        transcript.append(message)
        print(respond(message))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output.txt"),
        help="Where to save the conversation transcript.",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args().output)


if __name__ == "__main__":
    main()
