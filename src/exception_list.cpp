//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception_list.hpp>

#include <set>

namespace hpx
{
    error_code throws;        // "throw on error" special error_code;
                              //  note that it doesn't matter if this
                              //  isn't initialized before use since
                              //  the only use is to take its
                              //  address for comparison purposes

    exception_list::exception_list()
    {}

    exception_list::exception_list(boost::system::system_error const& e)
    {
        add(e);
    }

    void exception_list::add(boost::system::system_error const& e)
    {
        exceptions_.push_back(e);
    }

    boost::system::error_code exception_list::get_error() const
    {
        if (exceptions_.empty())
            return hpx::no_success;
        return exceptions_.front().code();
    }

    std::string indent_message(char const* msg_)
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

    std::string exception_list::get_message() const
    {
        if (exceptions_.empty())
            return "";

        if (1 == exceptions_.size())
            return exceptions_.front().what();

        std::string result("\n");

        exception_list_type::const_iterator end = exceptions_.end();
        exception_list_type::const_iterator it = exceptions_.begin();
        for (/**/; it != end; ++it) {
            result += "  ";
            result += indent_message((*it).what());
            if (result.find_last_of("\n") < result.size()-1)
                result += "\n";
        }
        return result;
    }

    std::size_t exception_list::get_error_count() const
    {
        return exceptions_.size();
    }

}
