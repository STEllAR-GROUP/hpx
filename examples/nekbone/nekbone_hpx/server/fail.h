#ifndef FAIL_H
#define FAIL_H

#if !defined(NAME_H)
#warning "fail.h" requires "name.h"
#endif

#define  die        PREFIXED_NAME( die       )
#define vdiagnostic PREFIXED_NAME(vdiagnostic)
#define  diagnostic PREFIXED_NAME( diagnostic)
#define vfail       PREFIXED_NAME(vfail      )
#define  fail       PREFIXED_NAME( fail      )

#ifdef __GNUC__
#  define ATTRBD   __attribute__ ((noreturn))
#  define ATTRB4V  __attribute__ ((format(printf,4,0)))
#  define ATTRB4   __attribute__ ((format(printf,4,5)))
#  define ATTRB4DV __attribute__ ((noreturn,format(printf,4,0)))
#  define ATTRB4D  __attribute__ ((noreturn,format(printf,4,5)))
#else
#  define ATTRBD
#  define ATTRB4V
#  define ATTRB4
#  define ATTRB4DV
#  define ATTRB4D
#endif

#define DEF_FUNS() \
  void  die(int status) ATTRBD; \
  void  diagnostic(const char *prefix, const char *file, unsigned line, \
                   const char *fmt, ...) ATTRB4  ; \
  void  fail      (int status,         const char *file, unsigned line, \
                   const char *fmt, ...) ATTRB4D ;
#define VDEF_FUNS() \
  void vdiagnostic(const char *prefix, const char *file, unsigned line, \
                   const char *fmt, va_list ap) ATTRB4V  ; \
  void vfail      (int status,         const char *file, unsigned line, \
                   const char *fmt, va_list ap) ATTRB4DV ;
DEF_FUNS()
#ifdef va_arg
VDEF_FUNS()
#endif

#undef VDEF_FUNS
#undef DEF_FUNS
#undef ATTRB4D
#undef ATTRB4DV
#undef ATTRB4
#undef ATTRB4V
#undef ATTRBD

#endif
