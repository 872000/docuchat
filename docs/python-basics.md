# Python Basics — Notes

## Variables and types

Python is dynamically typed, so you do not declare a variable's type. Assign a value with `=` and the type follows the value: `count = 3` makes an integer, `name = "Ada"` makes a string, and `prices = [1.5, 2.0]` makes a list.

## Functions

Define a function with the `def` keyword. Functions can take parameters and return values with `return`. If a function does not return anything explicitly, it returns `None`.

```python
def greet(name):
    return f"Hello, {name}!"
```

## Loops

A `for` loop iterates over any sequence, such as a list or a range of numbers. A `while` loop repeats as long as its condition stays true. Use `break` to exit a loop early and `continue` to skip to the next iteration.

## Dictionaries

A dictionary maps keys to values and is written with curly braces, for example `{"name": "Ada", "age": 36}`. Look a value up with square brackets: `person["name"]` gives `"Ada"`. Use the `.get()` method when a key might be missing, because it returns `None` instead of raising an error.
