MODULE_TOPDIR = ../..

PGM = i.sar.temporal_coherence

LIBES = $(RASTERLIB) $(GISLIB) $(DATETIMELIB) $(PARSONLIB) $(MATHLIB)
DEPENDENCIES = $(RASTERDEP) $(GISDEP) $(DATETIMEDEP) $(PARSONDEP)
EXTRA_INC = $(OCLINCPATH)
EXTRA_LIBS = $(OCLLIB)

include $(MODULE_TOPDIR)/include/Make/Module.make

default: cmd

# The OpenCL kernel is embedded in the binary as a C string literal.
$(OBJDIR)/ocl.o: tcoh_cl.h

tcoh_cl.h: tcoh.cl
	sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/^/"/' -e 's/$$/\\n"/' $< > $@
