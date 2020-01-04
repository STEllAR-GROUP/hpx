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

#ifndef JT28092007_destination_file_HPP_DEFINED
#define JT28092007_destination_file_HPP_DEFINED

#include <hpx/config.hpp>
#include <hpx/logging/manipulator.hpp>
#include <hpx/logging/message.hpp>

#include <boost/config.hpp>
#include <boost/smart_ptr/detail/spinlock.hpp>

#include <fstream>
#include <memory>
#include <mutex>
#include <string>

namespace hpx { namespace util { namespace logging { namespace destination {

    /**
    @brief settings for when constructing a file class. To see how it's used,
    see @ref dealing_with_flags.
*/
    struct file_settings
    {
        file_settings()
          : flush_each_time(true)
          , initial_overwrite(false)
          , do_append(true)
          , extra_flags(std::ios_base::out)
        {
        }

        /// if true (default), flushes after each write
        bool flush_each_time : 1;
        // if true it initially overwrites the file; default = false
        bool initial_overwrite : 1;
        // if true (default), opens the file for appending
        bool do_append : 1;
        /// just in case you have some extra flags to pass, when opening the file
        std::ios_base::openmode extra_flags;
    };

    namespace detail {
        inline std::ios_base::openmode open_flags(file_settings fs)
        {
            std::ios_base::openmode flags = std::ios_base::out;
            flags |= fs.extra_flags;
            if (fs.do_append && !fs.initial_overwrite)
                flags |= std::ios_base::app;
            if (fs.initial_overwrite)
                flags |= std::ios_base::trunc;
            return flags;
        }
    }    // namespace detail

    /**
    @brief Writes the string to a file
*/
    struct file : manipulator
    {
        HPX_NON_COPYABLE(file);

        typedef boost::detail::spinlock mutex_type;

        /**
        @brief constructs the file destination

        @param file_name name of the file
        @param set [optional] file settings - see file_settings class,
        and @ref dealing_with_flags
    */
        file(std::string const& file_name, file_settings set = file_settings())
          : name(file_name)
          , settings(set)
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
                out.open(name.c_str(), detail::open_flags(settings));
        }

        void close()
        {
            out.close();
        }

        std::string name;
        file_settings settings;
        mutable mutex_type mtx_ = BOOST_DETAIL_SPINLOCK_INIT;
        std::ofstream out;
    };

}}}}    // namespace hpx::util::logging::destination

#endif
