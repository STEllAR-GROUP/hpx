//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2003-2007 Jonathan Turkanis

// See http://www.boost.org/libs/iostreams for documentation.

#pragma once

#include <hpx/config.hpp>
#include <hpx/iostream/config/defines.hpp>
#include <hpx/iostream/categories.hpp>
#include <hpx/iostream/operations.hpp>

#include <fstream>
#include <iosfwd>
#include <locale>
#include <memory>
#include <string>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::iostream {

    HPX_CXX_CORE_EXPORT template <typename Ch>
    class basic_file
    {
    public:
        using char_type = Ch;

        struct category
          : seekable_device_tag
          , closable_tag
          , localizable_tag
          , flushable_tag
        {
        };

        explicit basic_file(std::string const& path,
            std::ios_base::openmode mode = std::ios_base::in |
                std::ios_base::out,
            std::ios_base::openmode base_mode = std::ios_base::in |
                std::ios_base::out);

        std::streamsize read(char_type* s, std::streamsize n);
        bool putback(char_type c);
        std::streamsize write(char_type const* s, std::streamsize n);
        std::streampos seek(stream_offset off, std::ios_base::seekdir way,
            std::ios_base::openmode which = std::ios_base::in |
                std::ios_base::out);

        void open(std::string const& path,
            std::ios_base::openmode mode = std::ios_base::in |
                std::ios_base::out,
            std::ios_base::openmode base_mode = std::ios_base::in |
                std::ios_base::out);

        [[nodiscard]] bool is_open() const;
        void close();
        bool flush();
        void imbue(std::locale const& loc)
        {
            pimpl_->file_.pubimbue(loc);
        }

    private:
        struct impl
        {
            impl(std::string const& path, std::ios_base::openmode mode)
            {
                file_.open(path.c_str(), mode);
            }

            ~impl()
            {
                if (file_.is_open())
                    file_.close();
            }

            impl(impl const&) = delete;
            impl(impl&) = delete;
            impl& operator=(impl const&) = delete;
            impl& operator=(impl&) = delete;

            std::basic_filebuf<Ch> file_;
        };

        std::shared_ptr<impl> pimpl_;
    };

    HPX_CXX_CORE_EXPORT using file = basic_file<char>;

#if defined(HPX_IOSTREAM_HAVE_WIDE_STREAMS)
    HPX_CXX_CORE_EXPORT using wfile = basic_file<wchar_t>;
#endif

    HPX_CXX_CORE_EXPORT template <typename Ch>
    struct basic_file_source : private basic_file<Ch>
    {
        using char_type = Ch;

        struct category
          : input_seekable
          , device_tag
          , closable_tag
        {
        };

        using basic_file<Ch>::read;
        using basic_file<Ch>::putback;
        using basic_file<Ch>::seek;
        using basic_file<Ch>::is_open;
        using basic_file<Ch>::close;

        explicit basic_file_source(std::string const& path,
            std::ios_base::openmode const mode = std::ios_base::in)
          : basic_file<Ch>(path, mode & ~std::ios_base::out, std::ios_base::in)
        {
        }

        void open(std::string const& path,
            std::ios_base::openmode const mode = std::ios_base::in)
        {
            basic_file<Ch>::open(
                path, mode & ~std::ios_base::out, std::ios_base::in);
        }
    };

    HPX_CXX_CORE_EXPORT using file_source = basic_file_source<char>;

#if defined(HPX_IOSTREAM_HAVE_WIDE_STREAMS)
    HPX_CXX_CORE_EXPORT using wfile_source = basic_file_source<wchar_t>;
#endif

    HPX_CXX_CORE_EXPORT template <typename Ch>
    struct basic_file_sink : private basic_file<Ch>
    {
        using char_type = Ch;

        struct category
          : output_seekable
          , device_tag
          , closable_tag
          , flushable_tag
        {
        };

        using basic_file<Ch>::write;
        using basic_file<Ch>::seek;
        using basic_file<Ch>::is_open;
        using basic_file<Ch>::close;
        using basic_file<Ch>::flush;

        explicit basic_file_sink(std::string const& path,
            std::ios_base::openmode const mode = std::ios_base::out)
          : basic_file<Ch>(path, mode & ~std::ios_base::in, std::ios_base::out)
        {
        }

        void open(std::string const& path,
            std::ios_base::openmode const mode = std::ios_base::out)
        {
            basic_file<Ch>::open(
                path, mode & ~std::ios_base::in, std::ios_base::out);
        }
    };

    HPX_CXX_CORE_EXPORT using file_sink = basic_file_sink<char>;

#if defined(HPX_IOSTREAM_HAVE_WIDE_STREAMS)
    HPX_CXX_CORE_EXPORT using wfile_sink = basic_file_sink<wchar_t>;
#endif

    //------------------Implementation of basic_file------------------------------//
    template <typename Ch>
    basic_file<Ch>::basic_file(std::string const& path,
        std::ios_base::openmode const mode,
        std::ios_base::openmode const base_mode)
    {
        open(path, mode, base_mode);
    }

    template <typename Ch>
    std::streamsize basic_file<Ch>::read(char_type* s, std::streamsize n)
    {
        std::streamsize result = pimpl_->file_.sgetn(s, n);
        return result != 0 ? result : -1;
    }

    template <typename Ch>
    bool basic_file<Ch>::putback(char_type c)
    {
        return !!pimpl_->file_.sputbackc(c);
    }

    template <typename Ch>
    std::streamsize basic_file<Ch>::write(char_type const* s, std::streamsize n)
    {
        return pimpl_->file_.sputn(s, n);
    }

    template <typename Ch>
    std::streampos basic_file<Ch>::seek(
        stream_offset off, std::ios_base::seekdir way, std::ios_base::openmode)
    {
        return iostream::seek(pimpl_->file_, off, way);
    }

    template <typename Ch>
    void basic_file<Ch>::open(std::string const& path,
        std::ios_base::openmode const mode,
        std::ios_base::openmode const base_mode)
    {
        pimpl_.reset(new impl(path, mode | base_mode));
    }

    template <typename Ch>
    bool basic_file<Ch>::is_open() const
    {
        return pimpl_->file_.is_open();
    }

    template <typename Ch>
    void basic_file<Ch>::close()
    {
        pimpl_->file_.close();
    }

    template <typename Ch>
    bool basic_file<Ch>::flush()
    {
        return pimpl_->file_.pubsync() == 0;
    }
}    // namespace hpx::iostream

#include <hpx/config/warnings_suffix.hpp>
