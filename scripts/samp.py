import argparse
import sys

from rich.console import Console
from rich.traceback import Traceback, install

# Set up Rich console and install traceback handler
console = Console()
install(show_locals=True, width=100, word_wrap=True)


def divide_numbers(a, b):
    x = 42  # An extra local variable for demonstration
    return a / b


def access_list_item(lst, index):
    temp = lst.copy()  # An extra local variable for demonstration
    return lst[index]


def recursive_function(n):
    local_var = f"Recursion level: {n}"  # Local variable for demonstration
    if n == 0:
        raise ValueError("Reached zero!")
    return recursive_function(n - 1)


def main():
    parser = argparse.ArgumentParser(description="Demonstrate Rich error handling")
    parser.add_argument("error_type", choices=["division", "index", "recursion"], help="Type of error to demonstrate")
    args = parser.parse_args()

    try:
        if args.error_type == "division":
            result = divide_numbers(10, 0)
        elif args.error_type == "index":
            my_list = [1, 2, 3]
            result = access_list_item(my_list, 5)
        elif args.error_type == "recursion":
            result = recursive_function(10)
    except Exception:
        console.print(Traceback(show_locals=True))
        sys.exit(1)


if __name__ == "__main__":
    main()
