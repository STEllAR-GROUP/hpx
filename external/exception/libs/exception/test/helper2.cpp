//Copyright (c) 2006-2008 Emil Dotchevski and Reverge Studios, Inc.

//Distributed under the Boost Software License, Version 1.0. (See accompanying
//file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "helper2.hpp"
#include <boost/throw_exception.hpp>

namespace
boost
    {
    namespace
    exception_test
        {
        inline
        some_boost_exception::
        some_boost_exception( int x ):
            x_(x)
            {
            }

        some_boost_exception::
        ~some_boost_exception() throw()
            {
            }

        inline
        some_std_exception::
        some_std_exception( int x ):
            x_(x)
            {
            }

        some_std_exception::
        ~some_std_exception() throw()
            {
            }

        template <>
        void
        throw_test_exception<some_boost_exception>( int x )
            {
            boost::throw_exception( some_boost_exception(x) );
            }

        template <>
        void
        throw_test_exception<some_std_exception>( int x )
            {
            boost::throw_exception( some_std_exception(x) );
            }
        }
    }
