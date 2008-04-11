This sample is to see what happens if using 
- a DLL 
  - using the logger
- an EXE using the DLL
  - it uses the logger from the DLL
  - it also uses another logger, specific to the EXE


I've made it work on Windows. If you can test it on Linux, please send me the diffs, and I'll update. Thanks!


The log from the dll :
- writes to a file called dll.txt and a file dllexe.txt (shared with the EXE)

The log from the exe
- writes to a file called exe.txt and a file dllexe.txt (shared with the DLL)




The dll.txt should look like:

14:35.23 beginning of dll log
14:35.23 message from exe on log from DLL 4
14:35.23 message from exe on log from DLL (2) 5
14:35.23 message from dll 1
14:35.23 hello world from dll 2
14:35.23 end of dll log



The exe.txt should look like:

[1] beginning of exe log1
[2] message from exe - before init_logs on EXE2
[3] message from exe - after init_logs on EXE3
[4] end of exe log 6



The dllexe.txt should look like:

14:35.23 beginning of dll log
[1] beginning of exe log1
[2] message from exe - before init_logs on EXE2
[3] message from exe - after init_logs on EXE3
14:35.23 message from exe on log from DLL 4
14:35.23 message from exe on log from DLL (2) 5
14:35.23 message from dll 1
14:35.23 hello world from dll 2
[4] end of exe log 6
14:35.23 end of dll log

