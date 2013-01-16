#ifndef C99_H
#define C99_H

#ifndef __STDC_VERSION__
#  define NO_C99
#elif __STDC_VERSION__ < 199901L
#  define NO_C99
#endif

#ifdef NO_C99
#  define restrict
#  define inline
#  undef NO_C99
#endif

#endif
