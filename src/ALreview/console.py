import os

from rich.console import Console
from rich.theme import Theme

here = os.path.dirname(os.path.abspath(__file__))
theme = Theme.read(os.path.join(here, "console.theme"))
console = Console(theme=theme)


def print_paper(paper):
    strip_text = lambda t: t.replace("\n", " ").replace("  ", " ").strip()

    print()
    console.print("[bold green]Title[/]: ", f"[u]{strip_text(paper['title'])}[/]")
    print()
    console.print(
        "[bold green]Abstract[/]: ",
        strip_text(paper["abstract"]),
    )
    print()
