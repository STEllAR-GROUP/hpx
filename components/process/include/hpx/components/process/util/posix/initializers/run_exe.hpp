// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PROCESS_POSIX_INITIALIZERS_RUN_EXE_HPP
#define HPX_PROCESS_POSIX_INITIALIZERS_RUN_EXE_HPP

#include <hpx/config.hpp>

#if !defined(HPX_WINDOWS)
#include <hpx/components/process/util/posix/initializers/initializer_base.hpp>
#include <hpx/runtime/serialization/string.hpp>

#include <boost/filesystem.hpp>
#include <boost/shared_array.hpp>

#include <string>

namespace hpx { namespace components { namespace process { namespace posix {

namespace initializers {

class run_exe_ : public initializer_base
{
public:
    run_exe_()
    {
        cmd_line_[0] = cmd_line_[1] = nullptr;
    }

    explicit run_exe_(const std::string &s)
      : s_(s)
    {
        cmd_line_[0] = const_cast<char*>(s_.c_str());
        cmd_line_[1] = nullptr;
    }

    template <class PosixExecutor>
    void on_exec_setup(PosixExecutor &e) const
    {
        e.exe = s_.c_str();
        if (!e.cmd_line)
            e.cmd_line = const_cast<char**>(cmd_line_);
    }

private:
    friend class hpx::serialization::access;

    template <typename Archive>
    void save(Archive& ar, unsigned const) const
    {
        ar & s_;
    }

    template <typename Archive>
    void load(Archive& ar, const unsigned int)
    {
        ar & s_;

        cmd_line_[0] = const_cast<char*>(s_.c_str());
        cmd_line_[1] = nullptr;
    }

    HPX_SERIALIZATION_SPLIT_MEMBER()

    std::string s_;
    char* cmd_line_[2];
};

inline run_exe_ run_exe(const char *s)
{
    return run_exe_(s);
}

inline run_exe_ run_exe(const std::string &s)
{
    return run_exe_(s);
}

inline run_exe_ run_exe(const boost::filesystem::path &p)
{
    return run_exe_(p.string());
}

}

}}}}

#endif
#endif
