# Copyright (c) 2011 Matthew Anderson
# Makefile
########
#ARCH = `uname -s` . `uname -m`
#ARCH = Linux.i686
#SUNDIALS_DIR = /home/astro/sundials
########
# Some stuff that's the same for all programs and machines 
########

# The commands to call the C and C++ compilers
CC = gcc 
CXX = g++ 
F90 = gfortran

# Where to find the include files:
INCLUDE = /home/manderson/rnpl/include

# library directory
LIB_DIR = /home/manderson/rnpl/lib

#LDLIBS = -lida.$(ARCH) -lshared.$(ARCH) -lnvecserial.$(ARCH) -lm
#LDLIBS = -lcvodes.$(ARCH) -lshared.$(ARCH) -lnvecserial.$(ARCH) -lm -lhdf5
LDLIBS = -lbbhutil -lgfortran 

CFLAGS = -g  -I$(INCLUDE) -L$(LIB_DIR)
########
# Stuff that depends on the specific program we're compiling
# To use this file for other programs, you'll need to change things
# in this section.
########

# The name of the output program 
PROG = mytest

# Object files that go into the final executable
OBJS = test.o parse.o util.o level.o

# Header files 
HDRS =

########
# Finally, the commands that actually make stuff
# You shouldn't have to change anything here 
# These commands are of the form:
#   output_thing:  stuff it depends on
#   <TAB> command to make output_thing 
########

# re-link the program when the object files change
$(PROG):  $(OBJS)
	$(CXX) $(CFLAGS) $(OBJS) $(LDLIBS) -o $(PROG) 

clean:
	rm -f test *.o *.sdf *.matlab *.idl core* $(PROG)
dataclean:
	rm -f *.o *.sdf *.matlab *.idl core*

# change a C object file whenever the corresponding source file
# or any of the header files changes
%.o : %.c $(HDRS) 
	$(CC) -c $(CFLAGS) $<
# What this means: 
# The %.o and %.c mean any .o file and the corresponding .c file
# All .c's are recompiled if you change one of the header files
# The $< symbol refers to the FIRST dependancy - so don't try to
# put $(HDRS) before %.c!

# change a C++ object file whenever the corresponding source file
# or any of the header files changes
%.o : %.cpp $(HDRS) 
	$(CXX) -c $(CFLAGS) $<
%.o : %.f $(HDRS) 
	$(F90) -c $<
