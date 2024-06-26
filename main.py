"""Over Under Tree Points: Main.py

Module Description
==================
This module is what the TA needs to see- our methods for accuracy/testing, and furthermore the program itself.

Copyright and Usage Information
===============================

This file is provided solely for the personal and use of Adam.
All forms of distribution of this code, whether as given or with any changes, are
expressly prohibited, unless permitted by either Adam or Grant.
For more information on copyright on this material,
please message me at adam.petrovic2005@gmail.com

This file is Copyright (c) 2024 Adam Petrovic
"""

import turtle
from app import App
from train import get_accuracy, model_accuracy


#  Runs the app
def run_app() -> None:
    """
    Runs the app
    """
    app = App()
    app.mainloop()



def betting_model_accuracy() -> None:
    """
    Testing the implementation of our algorithm
    """
    print(get_accuracy())


def implementation_accuracy() -> None:
    """
    Feature: Checks the accuracy of our betting model (Real Bets)
    """
    print(model_accuracy())


# if __name__ == "__main__":
#     import python_ta
#
#     python_ta.check_all(config={
#         'extra-imports': ['train', 'app'],  # the names (strs) of imported modules
#         'allowed-io': ['betting_model_accuracy', 'implementation_accuracy'],
#         'max-line-length': 120
#     })
