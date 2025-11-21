import os
import kagglehub

# -----------------------------
# 1. Download dataset
# -----------------------------
download_path = kagglehub.dataset_download("msambare/fer2013")
print("Downloaded to:", download_path)


# -----------------------------
# 2. Ensure .env exists in parent directory
# -----------------------------
env_path = os.path.join("..", ".env")

if not os.path.exists(env_path):
    with open(env_path, "w") as f:
        f.write("# Auto-generated .env file\n")
    print("Created new .env file at:", env_path)


# -----------------------------
# 3. Add or replace FER2013_PATH
# -----------------------------
with open(env_path, "r") as f:
    lines = f.readlines()

# Filter out old variable if re-running script
lines = [line for line in lines if not line.startswith("FER2013_PATH=")]

# Add new updated line
lines.append(f"FER2013_PATH={download_path}\n")

with open(env_path, "w") as f:
    f.writelines(lines)

print("FER2013_PATH saved to .env successfully.")
