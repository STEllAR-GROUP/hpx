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

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(push)
#pragma warning(disable : 4355)
#endif

#include <hpx/logging/detail/fwd.hpp>
#include <hpx/logging/detail/manipulator.hpp>
#include <hpx/logging/format/destination/convert_destination.hpp>

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
        typedef ::hpx::util::logging::detail::flag<file_settings> flag;

        file_settings()
          : flush_each_time(this, true)
          , initial_overwrite(this, false)
          , do_append(this, true)
          , extra_flags(this, std::ios_base::out)
        {
        }

        /// if true (default), flushes after each write
        flag::t<bool> flush_each_time;
        // if true it initially overwrites the file; default = false
        flag::t<bool> initial_overwrite;
        // if true (default), opens the file for appending
        flag::t<bool> do_append;

        /// just in case you have some extra flags to pass, when opening the file
        flag::t<std::ios_base::openmode> extra_flags;
    };

    namespace detail {
        inline std::ios_base::openmode open_flags(file_settings fs)
        {
            std::ios_base::openmode flags = std::ios_base::out;
            flags |= fs.extra_flags();
            if (fs.do_append() && !fs.initial_overwrite())
                flags |= std::ios_base::app;
            if (fs.initial_overwrite())
                flags |= std::ios_base::trunc;
            return flags;
        }

        struct file_info
        {
            file_info(const std::string& name_, file_settings const& settings_)
              : name(name_)
              ,
              //               out( new std::ofstream
              //                   ( name_.c_str(), open_flags(settings_) )),
              settings(settings_)
            {
            }

            void open()
            {
                out.reset(
                    new std::ofstream(name.c_str(), open_flags(settings)));
            }

            void close()
            {
                out.reset();
            }

            std::string name;
            std::shared_ptr<std::ofstream> out;
            file_settings settings;
        };
    }    // namespace detail

    /**
    @brief Writes the string to a file
*/
    struct file
      : is_generic
      , non_const_context<detail::file_info>
    {
        typedef non_const_context<detail::file_info> non_const_context_base;
        typedef boost::detail::spinlock mutex_type;

        /**
        @brief constructs the file destination

        @param file_name name of the file
        @param set [optional] file settings - see file_settings class,
        and @ref dealing_with_flags
    */
        file(const std::string& file_name, file_settings set = file_settings())
          : non_const_context_base(file_name, set)
        {
        }

        void operator()(const msg_type& msg) const
        {
            std::lock_guard<mutex_type> l(mtx_);

            if (!non_const_context_base::context().out)
                non_const_context_base::context()
                    .open();    // make sure file is opened
            convert::write(msg, *(non_const_context_base::context().out));
            if (non_const_context_base::context().settings.flush_each_time())
                non_const_context_base::context().out->flush();
        }

        bool operator==(const file& other) const
        {
            return non_const_context_base::context().name ==
                other.context().name;
        }

        /** configure through script
        right now, you can only specify the file name
    */
        void configure(const std::string& str)
        {
            // configure - the file name, for now
            non_const_context_base::context().close();
            non_const_context_base::context().name.assign(
                str.begin(), str.end());
        }

        static mutex_type mtx_;
    };

}}}}    // namespace hpx::util::logging::destination

#if defined(HPX_MSVC_WARNING_PRAGMA)
#pragma warning(pop)
#endif

#endif
