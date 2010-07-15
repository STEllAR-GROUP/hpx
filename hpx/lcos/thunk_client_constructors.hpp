//  Copyright (c) 2010-2011 Dylan Stark, Phillip LeBlanc
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_PP_IS_ITERATING

#if !defined(HPX_LCOS_THUNK_CLIENT_CONSTRUCTORS_JUN_27_2008_0440PM)
#define HPX_LCOS_THUNK_CLIENT_CONSTRUCTORS_JUN_27_2008_0440PM

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repeat.hpp>
#include <boost/preprocessor/iterate.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

#define BOOST_PP_ITERATION_PARAMS_1                                           \
    (3, (2, HPX_ACTION_ARGUMENT_LIMIT,                                        \
        "hpx/lcos/thunk_client_constructors.hpp"))                            \
            /**/
                
#include BOOST_PP_ITERATE()

#endif

///////////////////////////////////////////////////////////////////////////////
//  Preprocessor vertical repetition code
///////////////////////////////////////////////////////////////////////////////
#else // defined(BOOST_PP_IS_ITERATING)

#define N BOOST_PP_ITERATION()

  template <BOOST_PP_ENUM_PARAMS(N, typename Arg)>
  thunk_client(naming::id_type target,
      BOOST_PP_ENUM_BINARY_PARAMS(N, Arg, arg))
    : impl_(new wrapping_type(new wrapped_type(target,
          BOOST_PP_ENUM_PARAMS(N, arg))))
  {
    (*impl_)->set_gid(get_gid());
  }

#undef N

#endif
