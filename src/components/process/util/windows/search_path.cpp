// Copyright (c) 2006, 2007 Julio M. Merino Vidal
// Copyright (c) 2008 Ilya Sokolov, Boris Schaeling
// Copyright (c) 2009 Boris Schaeling
// Copyright (c) 2010 Felipe Tanus, Boris Schaeling
// Copyright (c) 2011, 2012 Jeff Flinn, Boris Schaeling
// Copyright (c) 2016 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_WINDOWS)
#include <hpx/exception.hpp>
#include <hpx/components/process/util/windows/search_path.hpp>

#include <boost/filesystem.hpp>
#include <boost/tokenizer.hpp>
#include <boost/array.hpp>
#include <boost/system/error_code.hpp>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <shellapi.h>

namespace hpx { namespace components { namespace process { namespace windows
{
#if defined(_UNICODE) || defined(UNICODE)
    std::wstring search_path(const std::wstring &filename, std::wstring path)
    {
        if (path.empty())
        {
            path = ::_wgetenv(L"PATH");
            if (path.empty())
            {
                HPX_THROW_EXCEPTION(invalid_status,
                    "process::search_path",
                    "Environment variable PATH not found");
            }
        }

        typedef boost::tokenizer<boost::char_separator<wchar_t>,
            std::wstring::const_iterator, std::wstring> tokenizer;
        boost::char_separator<wchar_t> sep(L";");
        tokenizer tok(path, sep);
        for (tokenizer::iterator it = tok.begin(); it != tok.end(); ++it)
        {
            boost::filesystem::path p = *it;
            p /= filename;
            boost::array<std::wstring, 4> extensions =
                { L"", L".exe", L".com", L".bat" };
            for (boost::array<std::wstring, 4>::iterator it2 = extensions.begin();
                it2 != extensions.end(); ++it2)
            {
                boost::filesystem::path p2 = p;
                p2 += *it2;
                boost::system::error_code ec;
                bool file = boost::filesystem::is_regular_file(p2, ec);
                if (!ec && file &&
                    SHGetFileInfoW(p2.c_str(), 0, 0, 0, SHGFI_EXETYPE))
                {
                    return p2.wstring();
                }
            }
        }
        return L"";
    }
#else
    std::string search_path(const std::string &filename, std::string path)
    {
        if (path.empty())
        {
            path = ::getenv("PATH");
            if (path.empty())
            {
                HPX_THROW_EXCEPTION(invalid_status,
                    "process::search_path",
                    "Environment variable PATH not found");
            }
        }

        typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
        boost::char_separator<char> sep(";");
        tokenizer tok(path, sep);
        for (tokenizer::iterator it = tok.begin(); it != tok.end(); ++it)
        {
            boost::filesystem::path p = *it;
            p /= filename;
            boost::array<std::string, 4> extensions =
                { "", ".exe", ".com", ".bat" };
            for (boost::array<std::string, 4>::iterator it2 = extensions.begin();
                it2 != extensions.end(); ++it2)
            {
                boost::filesystem::path p2 = p;
                p2 += *it2;
                boost::system::error_code ec;
                bool file = boost::filesystem::is_regular_file(p2, ec);
                if (!ec && file &&
                    SHGetFileInfoA(p2.string().c_str(), 0, 0, 0, SHGFI_EXETYPE))
                {
                    return p2.string();
                }
            }
        }
        return "";
    }
#endif
}}}}

#endif
