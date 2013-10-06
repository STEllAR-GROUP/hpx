/*=============================================================================
    Copyright (c) 2011 Thomas Heller

    Distributed under the Boost Software License, Version 1.0. (See accompanying
    file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
==============================================================================*/

#ifndef HPX_CONFIG_PP_ROUND_UP_ADD3_HPP
#define HPX_CONFIG_PP_ROUND_UP_ADD3_HPP

#include <boost/preprocessor/cat.hpp>

#define HPX_PP_ROUND_UP_ADD3(N)                                                    \
      BOOST_PP_CAT(HPX_PP_DO_ROUND_UP_ADD3_, N)()                                  \
/**/

#define HPX_PP_DO_ROUND_UP_ADD3_0()   5
#define HPX_PP_DO_ROUND_UP_ADD3_1()   5
#define HPX_PP_DO_ROUND_UP_ADD3_2()   5
#define HPX_PP_DO_ROUND_UP_ADD3_3()  10
#define HPX_PP_DO_ROUND_UP_ADD3_4()  10
#define HPX_PP_DO_ROUND_UP_ADD3_5()  10
#define HPX_PP_DO_ROUND_UP_ADD3_6()  10
#define HPX_PP_DO_ROUND_UP_ADD3_7()  10
#define HPX_PP_DO_ROUND_UP_ADD3_8()  15
#define HPX_PP_DO_ROUND_UP_ADD3_9()  15
#define HPX_PP_DO_ROUND_UP_ADD3_10() 15
#define HPX_PP_DO_ROUND_UP_ADD3_11() 15
#define HPX_PP_DO_ROUND_UP_ADD3_12() 15
#define HPX_PP_DO_ROUND_UP_ADD3_13() 20
#define HPX_PP_DO_ROUND_UP_ADD3_14() 20
#define HPX_PP_DO_ROUND_UP_ADD3_15() 20
#define HPX_PP_DO_ROUND_UP_ADD3_16() 20
#define HPX_PP_DO_ROUND_UP_ADD3_17() 20
#define HPX_PP_DO_ROUND_UP_ADD3_18() 30
#define HPX_PP_DO_ROUND_UP_ADD3_19() 30
#define HPX_PP_DO_ROUND_UP_ADD3_20() 30
// #define HPX_PP_DO_ROUND_UP_ADD3_21() 30
// #define HPX_PP_DO_ROUND_UP_ADD3_22() 30
// #define HPX_PP_DO_ROUND_UP_ADD3_23() 30
// #define HPX_PP_DO_ROUND_UP_ADD3_24() 30
// #define HPX_PP_DO_ROUND_UP_ADD3_25() 30
// #define HPX_PP_DO_ROUND_UP_ADD3_26() 30
// #define HPX_PP_DO_ROUND_UP_ADD3_27() 30
// #define HPX_PP_DO_ROUND_UP_ADD3_28() 40
// #define HPX_PP_DO_ROUND_UP_ADD3_29() 40
// #define HPX_PP_DO_ROUND_UP_ADD3_30() 40
// #define HPX_PP_DO_ROUND_UP_ADD3_31() 40
// #define HPX_PP_DO_ROUND_UP_ADD3_32() 40
// #define HPX_PP_DO_ROUND_UP_ADD3_33() 40
// #define HPX_PP_DO_ROUND_UP_ADD3_34() 40
// #define HPX_PP_DO_ROUND_UP_ADD3_35() 40
// #define HPX_PP_DO_ROUND_UP_ADD3_36() 40
// #define HPX_PP_DO_ROUND_UP_ADD3_37() 40
// #define HPX_PP_DO_ROUND_UP_ADD3_38() 50
// #define HPX_PP_DO_ROUND_UP_ADD3_39() 50
// #define HPX_PP_DO_ROUND_UP_ADD3_40() 50
// #define HPX_PP_DO_ROUND_UP_ADD3_41() 50
// #define HPX_PP_DO_ROUND_UP_ADD3_42() 50
// #define HPX_PP_DO_ROUND_UP_ADD3_43() 50
// #define HPX_PP_DO_ROUND_UP_ADD3_44() 50
// #define HPX_PP_DO_ROUND_UP_ADD3_45() 50
// #define HPX_PP_DO_ROUND_UP_ADD3_46() 50
// #define HPX_PP_DO_ROUND_UP_ADD3_47() 50

#endif
