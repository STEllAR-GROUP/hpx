// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PROCESS_POSIX_INITIALIZERS_SET_ARGS_HPP
#define HPX_PROCESS_POSIX_INITIALIZERS_SET_ARGS_HPP

#include <hpx/config.hpp>

#if !defined(HPX_WINDOWS)
#include <hpx/components/process/util/posix/initializers/initializer_base.hpp>
#include <hpx/runtime/serialization/string.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace hpx { namespace components { namespace process { namespace posix {

namespace initializers {

template <class Range>
class set_args_ : public initializer_base
{
public:
    set_args_()
    {
        args_.resize(1);
        args_[0] = nullptr;
    }

    explicit set_args_(const Range &args)
    {
        string_args_.resize(args.size());
        args_.resize(args.size() + 1);
        for (std::size_t i = 0; i != args.size(); ++i)
        {
            string_args_[i] = args[i];
            args_[i] = const_cast<char*>(string_args_[i].c_str());
        }
        args_[args.size()] = nullptr;
    }

    template <class PosixExecutor>
    void on_exec_setup(PosixExecutor &e) const
    {
        e.cmd_line = const_cast<char**>(args_.data());
        if (!e.exe && *args_[0])
            e.exe = args_[0];
    }

private:
    friend class hpx::serialization::access;

    template <typename Archive>
    void save(Archive& ar, unsigned const) const
    {
        ar & string_args_;
    }

    template <typename Archive>
    void load(Archive& ar, unsigned const)
    {
        ar & string_args_;

        args_.resize(string_args_.size() + 1);
        for (std::size_t i = 0; i != string_args_.size(); ++i)
        {
            args_[i] = const_cast<char*>(string_args_[i].c_str());
        }
        args_[string_args_.size()] = nullptr;
    }

    HPX_SERIALIZATION_SPLIT_MEMBER()

    std::vector<std::string> string_args_;
    std::vector<char*> args_;
};

template <class Range>
set_args_<Range> set_args(const Range &range)
{
    return set_args_<Range>(range);
}

}

}}}}

#endif
#endif
