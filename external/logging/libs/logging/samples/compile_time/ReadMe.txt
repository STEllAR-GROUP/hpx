Example of compile time.

Build all the .cpp files in debug mode.
Then, build them all, with BOOST_LOG_COMPILE_FAST_OFF directive, for ALL files.


Tested on 16 jan 2008/intel core duo 2.16Ghz machine, 5400Rpm HDD




VC 8.0
(no precompiled header)

Debug
Compile with BOOST_LOG_COMPILE_FAST_ON (default) - 33 secs 
Compile with BOOST_LOG_COMPILE_FAST_OFF  - 43 secs 

Release
Compile with BOOST_LOG_COMPILE_FAST_ON (default) - 24 secs 
Compile with BOOST_LOG_COMPILE_FAST_OFF - 29 secs	

gcc 3.4.2
Debug
Compile with BOOST_LOG_COMPILE_FAST_ON (default) - 24 secs 

Compile with BOOST_LOG_COMPILE_FAST_OFF  -  31 secs



gcc 4.1
Debug
Compile with BOOST_LOG_COMPILE_FAST_ON (default) - 20.5 secs 

Compile with BOOST_LOG_COMPILE_FAST_OFF  -  24 secs

