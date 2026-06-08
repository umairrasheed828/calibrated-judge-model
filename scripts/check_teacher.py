from src.teacher import get_completer


def main() -> None:
    complete = get_completer()
    print(complete("Reply with exactly: teacher online"))


if __name__ == "__main__":
    main()
