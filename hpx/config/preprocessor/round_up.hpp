/*=============================================================================
    Copyright (c) 2011 Thomas Heller

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
==============================================================================*/

#ifndef HPX_CONFIG_PP_ROUND_UP_HPP
#define HPX_CONFIG_PP_ROUND_UP_HPP

#include <boost/preprocessor/cat.hpp>

#define HPX_PP_ROUND_UP(N)                                                    \
      BOOST_PP_CAT(HPX_PP_DO_ROUND_UP_, N)()                                  \
/**/

#define HPX_PP_DO_ROUND_UP_0()   5
#define HPX_PP_DO_ROUND_UP_1()   5
#define HPX_PP_DO_ROUND_UP_2()   5
#define HPX_PP_DO_ROUND_UP_3()   5
#define HPX_PP_DO_ROUND_UP_4()   5
#define HPX_PP_DO_ROUND_UP_5()   5
#define HPX_PP_DO_ROUND_UP_6()  10
#define HPX_PP_DO_ROUND_UP_7()  10
#define HPX_PP_DO_ROUND_UP_8()  10
#define HPX_PP_DO_ROUND_UP_9()  10
#define HPX_PP_DO_ROUND_UP_10() 10
#define HPX_PP_DO_ROUND_UP_11() 15
#define HPX_PP_DO_ROUND_UP_12() 15
#define HPX_PP_DO_ROUND_UP_13() 15
#define HPX_PP_DO_ROUND_UP_14() 15
#define HPX_PP_DO_ROUND_UP_15() 15
#define HPX_PP_DO_ROUND_UP_16() 20
#define HPX_PP_DO_ROUND_UP_17() 20
#define HPX_PP_DO_ROUND_UP_18() 20
#define HPX_PP_DO_ROUND_UP_19() 20
#define HPX_PP_DO_ROUND_UP_20() 20
// #define HPX_PP_DO_ROUND_UP_21() 25
// #define HPX_PP_DO_ROUND_UP_22() 25
// #define HPX_PP_DO_ROUND_UP_23() 25
// #define HPX_PP_DO_ROUND_UP_24() 25
// #define HPX_PP_DO_ROUND_UP_25() 25
// #define HPX_PP_DO_ROUND_UP_26() 30
// #define HPX_PP_DO_ROUND_UP_27() 30
// #define HPX_PP_DO_ROUND_UP_28() 30
// #define HPX_PP_DO_ROUND_UP_29() 30
// #define HPX_PP_DO_ROUND_UP_30() 30
// #define HPX_PP_DO_ROUND_UP_31() 35
// #define HPX_PP_DO_ROUND_UP_32() 35
// #define HPX_PP_DO_ROUND_UP_33() 35
// #define HPX_PP_DO_ROUND_UP_34() 35
// #define HPX_PP_DO_ROUND_UP_35() 35
// #define HPX_PP_DO_ROUND_UP_36() 40
// #define HPX_PP_DO_ROUND_UP_37() 40
// #define HPX_PP_DO_ROUND_UP_38() 40
// #define HPX_PP_DO_ROUND_UP_39() 40
// #define HPX_PP_DO_ROUND_UP_40() 40
// #define HPX_PP_DO_ROUND_UP_41() 45
// #define HPX_PP_DO_ROUND_UP_42() 45
// #define HPX_PP_DO_ROUND_UP_43() 45
// #define HPX_PP_DO_ROUND_UP_44() 45
// #define HPX_PP_DO_ROUND_UP_45() 45
// #define HPX_PP_DO_ROUND_UP_46() 50
// #define HPX_PP_DO_ROUND_UP_47() 50
// #define HPX_PP_DO_ROUND_UP_48() 50
// #define HPX_PP_DO_ROUND_UP_49() 50
// #define HPX_PP_DO_ROUND_UP_50() 50

#endif
