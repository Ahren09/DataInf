
def print_colored(text, color='red'):
    foreground_colors = {
        'black': 30,
        'red': 31,
        'green': 32,
        'yellow': 33,
        'blue': 34,
        'magenta': 35,
        'cyan': 36,
        'white': 37,
    }
    print(f"\033[{foreground_colors[color]}m{text}\033[0m")
