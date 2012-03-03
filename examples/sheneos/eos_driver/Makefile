include ./make.inc

SOURCES=eosmodule.F90 readtable.F90 nuc_eos.F90 bisection.F90 findtemp.F90 findrho.F90 linterp_many.F90
FSOURCES=linterp.f

CLEANSTUFF=rm -rf *.o *.mod *.a

OBJECTS=$(SOURCES:.F90=.o)
FOBJECTS=$(FSOURCES:.f=.o)

EXTRADEPS=

MODINC=$(HDF5INCS)

all: nuc_eos.a driver onezone

onezone: nuc_eos.a one_zone.F90
	$(F90) $(F90FLAGS) -o one_zone one_zone.F90 nuc_eos.a $(HDF5LIBS)

driver: nuc_eos.a driver.F90
	$(F90) $(F90FLAGS) -o driver driver.F90 nuc_eos.a $(HDF5LIBS)

nuc_eos.a: $(OBJECTS) $(FOBJECTS)
	ar r nuc_eos.a *.o

$(OBJECTS): %.o: %.F90 $(EXTRADEPS)
	$(F90) $(F90FLAGS) $(DEFS) $(MODINC) -c $< -o $@

$(FOBJECTS): %.o: %.f $(EXTRADEPS)
	$(F90) $(F90FLAGS) $(DEFS) $(MODINC) -c $< -o $@


clean: 
	$(CLEANSTUFF)
