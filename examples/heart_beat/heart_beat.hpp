//  Copyright (c) 2010-2011 Dylan Stark
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(EXAMPLES_HEART_BEAT_JAN_29_2009_0917AM)
#define EXAMPLES_HEART_BEAT_JAN_29_2009_0917AM

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/plain_action.hpp>

using namespace hpx;

///////////////////////////////////////////////////////////////////////////////
int monitor(double frequency, double duration, double rate);

typedef
    actions::plain_result_action3<int, double, double, double, monitor>
monitor_action;

#endif
