//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/util/unlock_guard.hpp>

#include <boost/system/system_error.hpp>

#include <exception>
#include <mutex>
#include <set>
#include <string>
#include <utility>

namespace hpx
{
    namespace detail
    {
        std::string indent_message(std::string const& msg_)
        {
            std::string result;
            std::string msg(msg_);
            std::string::size_type pos = msg.find_first_of("\n");
            std::string::size_type first_non_ws = msg.find_first_not_of(" \n");
            std::string::size_type pos1 = 0;

            while (std::string::npos != pos) {
                if (pos > first_non_ws) {   // skip leading newline
                    result += msg.substr(pos1, pos-pos1+1);
                    pos = msg.find_first_of("\n", pos1 = pos+1);
                    if (std::string::npos != pos)
                        result += "  ";
                }
                else {
                    pos = msg.find_first_of("\n", pos1 = pos+1);
                }
            }

            result += msg.substr(pos1);
            return result;
        }
    }

    error_code throws;        // "throw on error" special error_code;
                              //
                              // Note that it doesn't matter if this isn't
                              // initialized before use since the only use is
                              // to take its address for comparison purposes.

    exception_list::exception_list()
      : hpx::exception(hpx::success)
    {}

    exception_list::exception_list(std::exception_ptr const& e)
      : hpx::exception(hpx::get_error(e), hpx::get_error_what(e))
    {
        add(e);
    }

    exception_list::exception_list(exception_list_type && l)
      : hpx::exception(l.size() ? hpx::get_error(l.front()) : success)
      , exceptions_(std::move(l))
    {}

    exception_list::exception_list(exception_list const& l)
      : hpx::exception(static_cast<hpx::exception const&>(l))
      , exceptions_(l.exceptions_)
    {}

    exception_list::exception_list(exception_list && l)
      : hpx::exception(std::move(static_cast<hpx::exception&>(l)))
      , exceptions_(std::move(l.exceptions_))
    {}

    exception_list& exception_list::operator=(exception_list const& l)
    {
        if (this != &l)
        {
            *static_cast<hpx::exception*>(this) = static_cast<hpx::exception const&>(l);
            exceptions_ = l.exceptions_;
        }
        return *this;
    }

    exception_list& exception_list::operator=(exception_list && l)
    {
        if (this != &l)
        {
            static_cast<hpx::exception&>(*this) =
                std::move(static_cast<hpx::exception&>(l));
            exceptions_ = std::move(l.exceptions_);
        }
        return *this;
    }

    ///////////////////////////////////////////////////////////////////////////
    boost::system::error_code exception_list::get_error() const
    {
        std::lock_guard<mutex_type> l(mtx_);
        if (exceptions_.empty())
            return hpx::no_success;
        return hpx::get_error(exceptions_.front());
    }

    std::string exception_list::get_message() const
    {
        std::lock_guard<mutex_type> l(mtx_);
        if (exceptions_.empty())
            return "";

        if (1 == exceptions_.size())
            return hpx::get_error_what(exceptions_.front());

        std::string result("\n");

        exception_list_type::const_iterator end = exceptions_.end();
        exception_list_type::const_iterator it = exceptions_.begin();
        for (/**/; it != end; ++it) {
            result += "  ";
            result += detail::indent_message(hpx::get_error_what(*it));
            if (result.find_last_of("\n") < result.size()-1)
                result += "\n";
        }
        return result;
    }

    void exception_list::add(std::exception_ptr const& e)
    {
        std::unique_lock<mutex_type> l(mtx_);
        if (exceptions_.empty())
        {
            hpx::exception ex;
            {
                util::unlock_guard<std::unique_lock<mutex_type> > ul(l);
                ex = hpx::exception(hpx::get_error(e));
            }

            // set the error code for our base class
            static_cast<hpx::exception&>(*this) = ex;
        }
        exceptions_.push_back(e);
    }
}
