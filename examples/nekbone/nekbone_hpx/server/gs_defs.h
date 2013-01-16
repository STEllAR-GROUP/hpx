#ifndef GS_DEFS_H
#define GS_DEFS_H

/* requires:
     <limits.h>, <float.h>   for GS_DEFINE_IDENTITIES()
     "types.h"               for gs_sint, gs_slong
*/
   
/*------------------------------------------------------------------------------
  Monoid Definitions
  
  Here are defined the domains and operations, each combination being a
  commutative semigroup, as well as the identity element making each a 
  commutative monoid.
------------------------------------------------------------------------------*/

/* the supported domains */
#define GS_FOR_EACH_DOMAIN(macro) \
  macro(double) \
  macro(float ) \
  macro(int   ) \
  macro(long  ) \
  WHEN_LONG_LONG(macro(long_long))
  
/* the supported ops */
#define GS_FOR_EACH_OP(T,macro) \
  macro(T,add) \
  macro(T,mul) \
  macro(T,min) \
  macro(T,max) \
  macro(T,bpr)

#define GS_DO_add(a,b) a+=b
#define GS_DO_mul(a,b) a*=b
#define GS_DO_min(a,b) if(b<a) a=b
#define GS_DO_max(a,b) if(b>a) a=b
#define GS_DO_bpr(a,b) \
  do if(b!=0) { uint a_ = a; uint b_ = b; \
       if(a_==0) { a=b_; break; } \
       for(;;) { if(a_<b_) b_>>=1; else if(b_<a_) a_>>=1; else break; } \
       a = a_; \
     } while(0)

/* the monoid identity elements */
#define GS_DEFINE_MONOID_ID(T,min,max) \
  static const T gs_identity_##T[] = { 0, 1, max, min, 0 };
#define GS_DEFINE_IDENTITIES() \
  GS_DEFINE_MONOID_ID(double, -DBL_MAX,  DBL_MAX) \
  GS_DEFINE_MONOID_ID(float , -FLT_MAX,  FLT_MAX) \
  GS_DEFINE_MONOID_ID(int   ,  INT_MIN,  INT_MAX) \
  GS_DEFINE_MONOID_ID(long  , LONG_MIN, LONG_MAX) \
  WHEN_LONG_LONG(GS_DEFINE_MONOID_ID(long_long,LLONG_MIN,LLONG_MAX))

/*------------------------------------------------------------------------------
  Enums and constants
------------------------------------------------------------------------------*/

/* domain enum */
#define LIST GS_FOR_EACH_DOMAIN(ITEM) gs_dom_n
#define ITEM(T) gs_##T,
typedef enum { LIST } gs_dom;
#undef ITEM
#undef LIST

#define gs_sint   TYPE_LOCAL(gs_int,gs_long,gs_long_long)
#define gs_slong TYPE_GLOBAL(gs_int,gs_long,gs_long_long)

/* domain type size array */
#define GS_DOM_SIZE_ITEM(T) sizeof(T),
#define GS_DEFINE_DOM_SIZES() \
  static const unsigned gs_dom_size[] = \
    { GS_FOR_EACH_DOMAIN(GS_DOM_SIZE_ITEM) 0 };

/* operation enum */
#define LIST GS_FOR_EACH_OP(T,ITEM) gs_op_n
#define ITEM(T,op) gs_##op,
typedef enum { LIST } gs_op;
#undef ITEM
#undef LIST

#endif
