#!/usr/bin/env python3
"""
Question Answering loop.
"""

qa_module = __import__('0-qa')
question_answer = qa_module.question_answer


def answer_loop(reference):
    """
    Interactive loop that answers questions.
    """
    exit_keywords = ['exit', 'quit', 'goodbye', 'bye']

    while True:
        user_input = input("Q: ").strip()

        if user_input.lower() in exit_keywords:
            print("A: Goodbye")
            break

        answer = question_answer(user_input, reference)

        if answer:
            print(f"A: {answer}")
        else:
            print("A: Sorry, I do not understand your question.")
