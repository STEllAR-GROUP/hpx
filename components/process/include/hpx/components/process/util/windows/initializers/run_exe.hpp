// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
// Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_WINDOWS)
#include <hpx/components/process/util/windows/initializers/initializer_base.hpp>
#include <hpx/modules/filesystem.hpp>
#include <hpx/serialization/string.hpp>

#include <string>

namespace hpx { namespace components { namespace process { namespace windows {

namespace initializers {

template <class String>
class run_exe_ : public initializer_base
{
public:
    run_exe_() {}
    explicit run_exe_(const String &s) : s_(s) {}

    template <class WindowsExecutor>
    void on_CreateProcess_setup(WindowsExecutor &e) const
    {
        e.exe = s_.c_str();
    }

private:
    friend class hpx::serialization::access;

    template <typename Archive>
    void serialize(Archive& ar, unsigned)
    {
        ar & s_;
    }

    String s_;
};

#if defined(_UNICODE) || defined(UNICODE)
inline run_exe_<std::wstring> run_exe(const wchar_t *ws)
{
    return run_exe_<std::wstring>(ws);
}

inline run_exe_<std::wstring> run_exe(const std::wstring &ws)
{
    return run_exe_<std::wstring>(ws);
}

inline run_exe_<std::wstring> run_exe(const filesystem::path &p)
{
    return run_exe_<std::wstring>(p.wstring());
}
#else
inline run_exe_<std::string> run_exe(const char *s)
{
    return run_exe_<std::string>(s);
}

inline run_exe_<std::string> run_exe(const std::string &s)
{
    return run_exe_<std::string>(s);
}

inline run_exe_<std::string> run_exe(const filesystem::path &p)
{
    return run_exe_<std::string>(p.string());
}
#endif

}

}}}}

#endif
