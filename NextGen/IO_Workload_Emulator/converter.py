#!/usr/bin/python
import sys
import shutil
import os
from random import random
from random import shuffle
logs = []
class Posix_IO:
  pass
def analyze_io(io):
  rank = io["rank"]
  operation = "R"
  num_io = 0
  num_seq = 0
  num_consec = 0
  io_start_timestamp = 0
  histobin = [0,0,0,0,0,0,0,0,0,0]
  binsize = [0,0,0,0,0,0,0,0,0,0]
  binsize[0] = 100
  binsize[1] = 1024
  binsize[2] = 10240
  binsize[3] = 102400
  binsize[4] = 1024 * 1024
  binsize[5] = 4 * 1024 * 1024
  binsize[6] = 10 * 1024 * 1024
  binsize[7] = 100 * 1024 * 1024
  binsize[8] = 1024 * 1024 * 1024
  binsize[9] = 10 * 1024 * 1024 * 1024

  if (rank >= 20) or (rank == -1):
    print "rank too large"
    return

  if rank != -1:
    curr_file = logs[rank]
    curr_file.write("." + io["path"] + " ")
    curr_file.write(str(int(io["POSIX_F_OPEN_START_TIMESTAMP"] * 1000000)) + " ")
    ioarray = []
    if int(io["POSIX_WRITES"]) != 0:
      operation = "W"
      num_io = int(io["POSIX_WRITES"])
      num_seq = int(io["POSIX_SEQ_WRITES"])
      num_consec = int(io["POSIX_CONSEC_WRITES"])
      io_start_timestamp = int(io["POSIX_F_WRITE_START_TIMESTAMP"] * 1000000)
      histobin[0] = int(io["POSIX_SIZE_WRITE_0_100"])
      histobin[1] = int(io["POSIX_SIZE_WRITE_100_1K"])
      histobin[2] = int(io["POSIX_SIZE_WRITE_1K_10K"])
      histobin[3] = int(io["POSIX_SIZE_WRITE_10K_100K"])
      histobin[4] = int(io["POSIX_SIZE_WRITE_100K_1M"])
      histobin[5] = int(io["POSIX_SIZE_WRITE_1M_4M"])
      histobin[6] = int(io["POSIX_SIZE_WRITE_4M_10M"])
      histobin[7] = int(io["POSIX_SIZE_WRITE_10M_100M"])
      histobin[8] = int(io["POSIX_SIZE_WRITE_100M_1G"])
      histobin[9] = int(io["POSIX_SIZE_WRITE_1G_PLUS"])
    else:
      operation = "R"
      num_io = int(io["POSIX_READS"])
      num_seq = int(io["POSIX_SEQ_READS"])
      num_consec = int(io["POSIX_CONSEC_READS"])
      io_start_timestamp = int(io["POSIX_F_READ_START_TIMESTAMP"] * 1000000)
      histobin[0] = int(io["POSIX_SIZE_READ_0_100"])
      histobin[1] = int(io["POSIX_SIZE_READ_100_1K"])
      histobin[2] = int(io["POSIX_SIZE_READ_1K_10K"])
      histobin[3] = int(io["POSIX_SIZE_READ_10K_100K"])
      histobin[4] = int(io["POSIX_SIZE_READ_100K_1M"])
      histobin[5] = int(io["POSIX_SIZE_READ_1M_4M"])
      histobin[6] = int(io["POSIX_SIZE_READ_4M_10M"])
      histobin[7] = int(io["POSIX_SIZE_READ_10M_100M"])
      histobin[8] = int(io["POSIX_SIZE_READ_100M_1G"])
      histobin[9] = int(io["POSIX_SIZE_READ_1G_PLUS"])
      print "R"
    # p for POSIX
    curr_file.write(str(num_io) + " p " + str(io_start_timestamp) + " ")
    freq_access_size = [0,0,0,0]
    freq_access_count = [0,0,0,0]
    freq_access_size[0] = int(io["POSIX_ACCESS1_ACCESS"])
    freq_access_size[1] = int(io["POSIX_ACCESS2_ACCESS"])
    freq_access_size[2] = int(io["POSIX_ACCESS3_ACCESS"])
    freq_access_size[3] = int(io["POSIX_ACCESS4_ACCESS"])
    freq_access_count[0] = int(io["POSIX_ACCESS1_COUNT"])
    freq_access_count[1] = int(io["POSIX_ACCESS2_COUNT"])
    freq_access_count[2] = int(io["POSIX_ACCESS3_COUNT"])
    freq_access_count[3] = int(io["POSIX_ACCESS4_COUNT"])
    num_noseq = num_io - num_seq
    num_seq -= num_consec
    #print histobin
    for i in range(4):
      if freq_access_count[i] == 0:
        break
      for j in range(9):
        if freq_access_size[i] < binsize[j]:
          histobin[j] -= freq_access_count[i]
          break
        histobin[9] -= freq_access_count[i]
      for k in range(freq_access_count[i]):
        ioarray.append(freq_access_size[i])
    #print histobin
    #print "ioarray"
    #print ioarray
    binsize_min = 0
    for i in range(10):
      if histobin[i] > 0:
        for j in range(histobin[i]):
          ioarray.append(int(random() * (binsize[i] - binsize_min)) + binsize_min)
    noseq_list = []
    for i in range(num_noseq):
      pick = int(random() * num_io)
      while pick in noseq_list:
        pick = int(random() * num_io)
      noseq_list.append(pick)
    noconsec_list = []
    for i in range(num_seq):
      pick = int(random() * num_io)
      while (pick in noseq_list) or (pick in noconsec_list):
        pick = int(random() * num_io)
      noconsec_list.append(pick)
    shuffle(ioarray)
    last_offset = int(random() * 1024)
    for i in range(num_io):
      if i in noseq_list:
        last_offset = int(random() * last_offset)
      if i in noconsec_list:
        last_offset += int(random() * 1024)
      if ioarray:
        #print str(i) + " " + str(len(ioarray)) + " " + str(num_io)
        curr_file.write("(" + operation + " " +  str(last_offset) + " " + str(ioarray[i]) + ") ")
        last_offset += ioarray[i]
    curr_file.write("\n")
  print curr_file
  
def parse_posix(fh):
  io = {}
  line = fh.readline()
  line_component = line.split("\t")
  io["rank"] = int(line_component[1])
  io["record_id"] = line_component[2]
  io_type = "Lustre"
  if "dvs" in line_component[7]:
    if "dws" in line_component[6]:
      io_type = "BB"
    else:
      io_type = "GPFS"
  io["type"] = io_type
  io["path"] = line_component[5].replace(line_component[6], "")
  x = fh.tell()
  while io["record_id"] in line:
    line_component = line.split("\t")
    io[line_component[3]] = float(line_component[4])
    x = fh.tell()
    line = fh.readline()
  fh.seek(x)
  analyze_io(io)
#  print(io)
filename = sys.argv[1].replace(".darshan.txt", "")
ofile = open(sys.argv[1])
nprocs = 1
curr_line = ofile.readline()
while "nprocs" not in curr_line:
  curr_line = ofile.readline()
  continue
nprocs = int(curr_line.split(" ")[2])
print sys.argv[1]
print sys.argv[1][:-12]
print filename
# remove later
if nprocs > 20:
  nprocs = 20
if os.path.isdir(filename):
  shutil.rmtree(filename)
os.mkdir(filename)
for i in range(nprocs):
  logs.append(open(filename + "/logs." + str(i), "w"))
while curr_line:
  if "POSIX_OPENS" in curr_line:
    parse_posix(ofile)
  x = ofile.tell()
  curr_line = ofile.readline()
ofile.seek(x)
