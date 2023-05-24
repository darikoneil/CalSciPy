from pathlib import Path

filename = Path.cwd().joinpath("rtd_requirements.txt")

with open(filename, "r+") as file:
    lines = file.readlines()
    file.seek(0)  # necessary for resetting pointer
    for line in lines:
        if "cupy" not in line and "CalSciPy" not in line:
            file.write(line)
    file.truncate()
