# Most logic is in regtest/util/Makefile.host.
# Setting exe-specific variables here.
# Assuming source files are in 'src' sub-dir, include files are
# in 'inc' sub-dir.

TARGET = host

SRCS = main_bnn_26.cpp

USES_NVIDIA = 1
USES_ACL_HOST_UTILS = 1
       
ifeq ($(SAVE_MATRIX),1)
CPPFLAGS += -DSAVE_MATRIX
endif

# Profiling
#ifeq ($(PROFILE),1)
#CPPFLAGS += -DGPU_PROFILING
#endif

#ifeq ($(DUMP),1)
#CPPFLAGS += -DDUMP_INPUTS_OUTPUTS
#endif

# can also set
#   PLATFORM = ACL|NVIDIA (default ACL)
#   DEBUG    = 0|1 (default 0)
#   LIBS
#CPPFLAGS += -O3
CPPFLAGS += -std=c++11 -O3 -march=native -ffast-math -fassociative-math -mfpmath=sse
#   LINKFLAGS

# Top-level regtest dir.
REG_LOCAL_ROOT_DIR_PATH ?= ../../../../..
#include $(REG_LOCAL_ROOT_DIR_PATH)/hld/opencl/util/Makefile.host
# Use this Makefile.host if compiling standalone (without ARC infrastructure)
include Makefile.host

