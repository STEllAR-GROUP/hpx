//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_EXCEPTION_LIST_OCT_06_2008_0942AM)
#define HPX_EXCEPTION_LIST_OCT_06_2008_0942AM

#include <list>
#include <string>
#include <hpx/exception.hpp>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXCEPTION_EXPORT exception_list
    {
    private:
        typedef std::list<boost::system::system_error> exception_list_type;
        exception_list_type exceptions_;

    public:
        exception_list();
        explicit exception_list(boost::system::system_error const& e);

        ///
        void add(boost::system::system_error const& e);

        ///
        boost::system::error_code get_error() const;

        ///
        std::string get_message() const;

        ///
        std::size_t get_error_count() const;
    };

}

#include <hpx/config/warnings_suffix.hpp>

#endif


