import sys
from importlib.metadata import version

pins = ["tensorflow", "keras", "numpy", "pillow", "flask", "flask-sqlalchemy",
        "sqlalchemy", "werkzeug", "python-dotenv"]
lines = []
for p in pins:
    name = "tensorflow-cpu" if p == "tensorflow" else p   # CPU-only build for the container
    lines.append(f"{name}=={version(p)}")
lines.append("psycopg2-binary>=2.9")                        # PostgreSQL driver
open("requirements-app.txt", "w").write("\n".join(lines) + "\n")
print("python", sys.version.split()[0])
print("\n".join(lines))