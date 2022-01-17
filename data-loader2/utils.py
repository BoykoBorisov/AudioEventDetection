import subprocess
from os.path import exists

blacklist = set(["/m/09x0r", "/m/04rlf"])
dest = "output_bal_train"

def is_blacklist_entry(entry):
  for label in entry:
    if label not in blacklist:
      return False
  return True

def download_file(csv_row):
  cmd = "ffmpeg -hide_banner -loglevel error -ss {start} -t 10 -i $(youtube-dl -f 'bestaudio' -g \"https://youtube.com/watch?v={id}\") -ar {fs} -- \"{dest}/{id}_{start}.wav\""
  cmd = cmd.format(
    start = str(csv_row[1].strip().strip(",")), 
    id = csv_row[0].strip(","), 
    dest = dest, 
    fs = str(16000))
  subprocess.call(cmd, shell=True)

def download_file_if_not_blacklised(row):
  # if not len(row) > 4 and not is_blacklist_entry(row[3].strip("\"").split(",")):
    path = "{dest}/{id}_{start}.wav".format(start = str(row[1].strip().strip(",")), id = row[0].strip(","), dest = dest)
    if not exists(path):
      download_file(row)