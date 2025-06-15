#!/usr/bin/env python3
"""
Basic QA loop
"""


def qa_loop():
    """
    Basic QA loop
    """
    farewells = ["exit", "quit", "goodbye", "bye"]

    user_input = ""
    while True:
        user_input = input("Q: ")

        if user_input.lower() in farewells:
            print("A: Goodbye")
            exit()

        print("A: ")


if __name__ == "__main__":
    qa_loop()
