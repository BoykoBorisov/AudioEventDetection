import subprocess

blacklist = set(["/m/09x0r", "/m/04rlf"])
dest = "output"

def is_blacklist_entry(entry):
  for label in entry:
    if label not in blacklist:
      return False
  return True

def download_file(csv_row):
  cmd = "ffmpeg -ss {start} -t 10 -i $(youtube-dl -f 'bestaudio' -g \"https://youtube.com/watch?v={id}\") -ar {fs} -- \"{dest}/{id}_{start}.wav\""
  cmd = cmd.format(
    start = str(csv_row[1].strip().strip(",")), 
    id = csv_row[0].strip(","), 
    dest = dest, 
    fs = str(16000))
  # print(cmd)
  # subprocess.call("youtube-dl", shell = True)
  subprocess.call(cmd, shell=True)

def download_file_if_not_blacklised(row):
  if not is_blacklist_entry(row[3].strip("\"").split(",")):
    download_file(row)