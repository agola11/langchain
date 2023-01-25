import subprocess
from pathlib import Path


def main() -> None:
    p = Path(__file__).absolute().parent / "docker-compose.yaml"
    subprocess.run(["docker-compose", "-f", str(p), "pull"])
    subprocess.run(["docker-compose", "-f", str(p), "up"])


if __name__ == "__main__":
    main()
