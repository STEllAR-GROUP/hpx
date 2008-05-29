//Copyright (c) 2006-2008 Emil Dotchevski and Reverge Studios, Inc.

//Distributed under the Boost Software License, Version 1.0. (See accompanying
//file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef UUID_BC765EB4CA2A11DCBDC5828355D89593
#define UUID_BC765EB4CA2A11DCBDC5828355D89593

#include <boost/exception/exception.hpp>
#include <exception>

namespace
boost
    {
    namespace
    exception_test
        {
        struct
        some_boost_exception:
            public boost::exception,
            public std::exception
            {
            explicit some_boost_exception( int x );
            virtual ~some_boost_exception() throw();
            int x_;
            };

        struct
        some_std_exception:
            public std::exception
            {
            explicit some_std_exception( int x );
            virtual ~some_std_exception() throw();
            int x_;
            };

        template <class>
        void throw_test_exception( int );

        template <>
        void throw_test_exception<some_boost_exception>( int );

        template <>
        void throw_test_exception<some_std_exception>( int );
        }
    }

#endif
