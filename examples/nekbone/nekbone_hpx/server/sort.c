#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "c99.h"
#include "name.h"
#include "fail.h"
#include "types.h"
#include "mem.h"

#define T unsigned int
#define SORT_SUFFIX _ui
#include "sort_imp.h"
#undef SORT_SUFFIX
#undef T

#if defined(USE_LONG) || defined(GLOBAL_LONG)
#  define T unsigned long
#  define SORT_SUFFIX _ul
#  include "sort_imp.h"
#  undef SORT_SUFFIX
#  undef T
#endif

#if defined(USE_LONG_LONG) || defined(GLOBAL_LONG_LONG)
#  define T unsigned long long
#  define SORT_SUFFIX _ull
#  include "sort_imp.h"
#  undef SORT_SUFFIX
#  undef T
#endif
