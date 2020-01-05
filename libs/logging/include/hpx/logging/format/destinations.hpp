// destination_defaults.hpp

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

#ifndef HPX_LOGGING_FORMAT_DESTINATIONS_HPP
#define HPX_LOGGING_FORMAT_DESTINATIONS_HPP

#include <hpx/config.hpp>
#include <hpx/logging/manipulator.hpp>

#include <ios>
#include <iosfwd>
#include <memory>
#include <ostream>
#include <string>

namespace hpx { namespace util { namespace logging { namespace destination {

    /**
    @brief Writes the string to console
*/
    struct cout : manipulator
    {
        static std::shared_ptr<cout> HPX_EXPORT make();

    protected:
        cout() = default;

        HPX_EXPORT ~cout();
    };

    /**
    @brief Writes the string to cerr
*/
    struct cerr : manipulator
    {
        static std::shared_ptr<cerr> HPX_EXPORT make();

    protected:
        cerr() = default;

        HPX_EXPORT ~cerr();
    };

    /**
    @brief writes to stream.

    @note:
    The stream must outlive this object! Or, clear() the stream,
    before the stream is deleted.
*/
    struct stream : manipulator
    {
        static std::shared_ptr<stream> HPX_EXPORT make(
            std::ostream* stream_ptr);

        /**
        @brief resets the stream. Further output will be written to this stream
    */
        void set_stream(std::ostream* stream_ptr)
        {
            ptr = stream_ptr;
        }

        /**
        @brief clears the stream. Further output will be ignored
    */
        void clear()
        {
            ptr = nullptr;
        }

    protected:
        explicit stream(std::ostream* stream_ptr)
          : ptr(stream_ptr)
        {
        }

        HPX_EXPORT ~stream();

    protected:
        std::ostream* ptr;
    };

    /**
    @brief Writes the string to output debug window

    For non-Windows systems, this is the console.
*/
    struct dbg_window : manipulator
    {
        static std::shared_ptr<dbg_window> HPX_EXPORT make();

    protected:
        dbg_window() = default;

        HPX_EXPORT ~dbg_window();
    };

    /**
    @brief Writes the string to a file
*/
    struct file : manipulator
    {
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

        /**
        @brief constructs the file destination

        @param file_name name of the file
        @param set [optional] file settings - see file_settings class,
        and @ref dealing_with_flags
    */
        static std::shared_ptr<file> HPX_EXPORT make(
            std::string const& file_name, file_settings set = {});

    protected:
        file(std::string const& file_name, file_settings set)
          : name(file_name)
          , settings(set)
        {
        }

        HPX_EXPORT ~file();

        std::string name;
        file_settings settings;
    };

}}}}    // namespace hpx::util::logging::destination

#endif /*HPX_LOGGING_FORMAT_DESTINATIONS_HPP*/
