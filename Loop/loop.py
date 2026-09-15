import os
import shutil
import subprocess
import time
from pathlib import Path

REFERENCES_DIR = Path("./next_token_predictor_references")

CLAUDE_HOME_DIR = Path("../.claude")
SCRATCH_DIR = Path("./scratch")

OPTIMIZING_FILE = Path("./next_token_predictor.py")

LOG_DIR = Path("./logs")
HISTORY_DIR = Path("./history")

LOG_DIR.mkdir(exist_ok=True)
HISTORY_DIR.mkdir(exist_ok=True)

LOOP_COUNTER_FILE = Path("loop_counter")

PROMPT = Path("PROMPT.md").read_text()

def run_ai_maybe_out_of_tokens():
  shutil.rmtree(SCRATCH_DIR, ignore_errors=True)
  SCRATCH_DIR.mkdir()
  result = subprocess.run([
    "systemd-run", "--user", "--scope", "--quiet",
    "-p", f"MemoryMax=7G", 
    "-p", f"CPUQuota=200%", 
    "-p", f"TasksMax=256", 
    "timeout", "-s", "KILL", str(60 * 20),
    "bwrap",
    "--unshare-all",
    "--share-net",
    "--die-with-parent",
    "--clearenv",
    "--setenv", "PATH", os.environ["PATH"],
    "--setenv", "HOME", str(SCRATCH_DIR.resolve()),
    "--setenv", "CLAUDE_CONFIG_DIR", str(CLAUDE_HOME_DIR),
    "--ro-bind", "/", "/",
    "--proc", "/proc",
    "--dev", "/dev",
    "--tmpfs", "/tmp",
    "--bind", str(CLAUDE_HOME_DIR.resolve()), str(CLAUDE_HOME_DIR.resolve()),
    "--bind", str(SCRATCH_DIR.resolve()), str(SCRATCH_DIR.resolve()),
    "--bind", str(OPTIMIZING_FILE.resolve()), str(OPTIMIZING_FILE.resolve()),
    "--bind", str(REFERENCES_DIR.resolve()), str(REFERENCES_DIR.resolve()),
    "claude",
    "-p", PROMPT,
    "--max-turns", str(50),
    "--output-format", "json",
  ], capture_output=True, text=True)
  return result

def run_ai(loop_count):
  result = None
  while True:
    result = run_ai_maybe_out_of_tokens()
    if result.stderr != "":
      print ("Got error, retrying in 10 minutes:", result.stderr)
      time.sleep(60 * 10)
    else:
      break
  (LOG_DIR / f"log{loop_count}").write_text(f"Returned with {result.returncode}\n\nStdout: {str(result.stdout)}\n\nStderr: {str(result.stderr)}\n")

def run_evaluate_subprocess():
  return subprocess.run([
    "systemd-run", "--user", "--scope", "--quiet",
    "-p", f"MemoryMax=7G", 
    "-p", f"CPUQuota=200%", 
    "-p", f"TasksMax=256", 
    "timeout", "-s", "KILL", str(60 * 5),
    "bwrap",
    "--unshare-all",
    "--die-with-parent",
    "--clearenv",
    "--setenv", "PATH", "/usr/bin:/bin",
    "--ro-bind", "/", "/",
    "--proc", "/proc",
    "--dev", "/dev",
    "python3", "./eval.py",
  ], capture_output=True, text=True)

def run_evaluate():
  eval_result = run_evaluate_subprocess()
  if eval_result.returncode == 0:
    eval_result_lines = eval_result.stdout.strip().splitlines()
    if len(eval_result_lines) == 1:
      return float(eval_result_lines[0])

  return 0

loop_count = 0
if LOOP_COUNTER_FILE.exists():
  loop_count = int(LOOP_COUNTER_FILE.read_text().splitlines()[-1].split(" ")[0]) + 1

def start_elapsed_tracking():
  return time.perf_counter()

def finish_elapsed_tracking(start):
  print (f"Finished in {time.perf_counter() - start:.3f}s")

while True:
  print (f"Loop count: {loop_count}")

  print ("Running AI")
  start = start_elapsed_tracking()
  run_ai(loop_count)
  finish_elapsed_tracking(start)

  print ("Evaluating")
  start = start_elapsed_tracking()
  curr_eval_score = run_evaluate()
  finish_elapsed_tracking(start)
  print ("Got: ", curr_eval_score)

  print ("Saving")
  start = start_elapsed_tracking()
  new_history_entry_dir = Path(HISTORY_DIR / ("entry_" + str(loop_count)))
  new_history_entry_dir.mkdir()
  shutil.copy2(OPTIMIZING_FILE, new_history_entry_dir)
  shutil.copytree(REFERENCES_DIR, new_history_entry_dir / REFERENCES_DIR.name)
  finish_elapsed_tracking(start)

  with open(LOOP_COUNTER_FILE, "w") as loop_counter_file:
    loop_counter_file.write(str(loop_count) + " " + str(curr_eval_score) + "\n")

  loop_count += 1
