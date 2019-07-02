// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
// Copyright (c) 2016 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PROCESS_WINDOWS_INITIALIZERS_SET_ARGS_HPP
#define HPX_PROCESS_WINDOWS_INITIALIZERS_SET_ARGS_HPP

#include <hpx/config.hpp>

#if defined(HPX_WINDOWS)
#include <hpx/components/process/util/windows/initializers/initializer_base.hpp>
#include <hpx/runtime/serialization/string.hpp>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/range/algorithm/copy.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>
#include <boost/shared_array.hpp>

#include <sstream>
#include <string>

namespace hpx { namespace components { namespace process { namespace windows {

namespace initializers {

template <class Range>
class set_args_ : public initializer_base
{
private:
    typedef typename Range::const_iterator ConstIterator;
    typedef typename Range::value_type String;
    typedef typename String::value_type Char;
    typedef std::basic_ostringstream<Char> OStringStream;

public:
    set_args_() {}

    explicit set_args_(const Range &args)
    {
        ConstIterator it = boost::const_begin(args);
        ConstIterator end = boost::const_end(args);
        if (it != end)
        {
            exe_ = *it;
            OStringStream os;
            for (; it != end; ++it)
            {
                if (boost::algorithm::contains(*it,
                    String(1, static_cast<Char>(' '))))
                {
                    os << static_cast<Char>('"') << *it <<
                        static_cast<Char>('"');
                }
                else
                {
                    os << *it;
                }
                os << static_cast<Char>(' ');
            }
            cmd_line_ = os.str();
        }
        else
        {
            exe_.clear();
            cmd_line_.clear();
        }
    }

    template <class WindowsExecutor>
    void on_CreateProcess_setup(WindowsExecutor &e) const
    {
        e.cmd_line = const_cast<char*>(cmd_line_.c_str());
        if (!e.exe && !exe_.empty())
            e.exe = exe_.c_str();
    }

private:
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        ar & exe_ & cmd_line_;
    }

    String exe_;
    String cmd_line_;
};

template <class Range>
set_args_<Range> set_args(Range const& range)
{
    return set_args_<Range>(range);
}

}

}}}}

#endif
#endif
