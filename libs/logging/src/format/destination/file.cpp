// destination_file.hpp

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
//  SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details

#include <hpx/logging/format/destinations.hpp>

#include <hpx/config.hpp>
#include <hpx/logging/message.hpp>

#include <boost/config.hpp>
#include <boost/smart_ptr/detail/spinlock.hpp>

#include <fstream>
#include <memory>
#include <mutex>
#include <string>

namespace hpx { namespace util { namespace logging { namespace destination {

    file::~file() = default;

    static std::ios_base::openmode open_flags(file::file_settings fs)
    {
        std::ios_base::openmode flags = std::ios_base::out;
        flags |= fs.extra_flags;
        if (fs.do_append && !fs.initial_overwrite)
            flags |= std::ios_base::app;
        if (fs.initial_overwrite)
            flags |= std::ios_base::trunc;
        return flags;
    }

    struct file_impl : file
    {
        typedef boost::detail::spinlock mutex_type;

        explicit file_impl(std::string const& file_name, file_settings set)
          : file(file_name, set)
        {
        }

        void operator()(const message& msg) override
        {
            std::lock_guard<mutex_type> l(mtx_);

            open();    // make sure file is opened
            out << msg.full_string();
            if (settings.flush_each_time)
                out.flush();
        }

        /** configure through script
        right now, you can only specify the file name
    */
        void configure(std::string const& str) override
        {
            // configure - the file name, for now
            close();
            name = str;
        }

    private:
        void open()
        {
            if (!out.is_open())
                out.open(name.c_str(), open_flags(settings));
        }

        void close()
        {
            out.close();
        }

        std::ofstream out;
        mutable mutex_type mtx_ = BOOST_DETAIL_SPINLOCK_INIT;
    };

    std::shared_ptr<file> file::make(
        std::string const& file_name, file_settings set)
    {
        return std::make_shared<file_impl>(file_name, set);
    }

}}}}    // namespace hpx::util::logging::destination
