//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_LOGGING)

#include <hpx/exception.hpp>
#include <hpx/runtime/naming_fwd.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/components/console_logging.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/static.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/logging/format/named_write.hpp>
#include <hpx/util/logging/format/destination/defaults.hpp>
#include <hpx/util/init_logging.hpp>

#include <boost/version.hpp>
#include <boost/config.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem/operations.hpp>

#include <boost/assign/std/vector.hpp>

#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#endif

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util
{
    typedef logging::writer::named_write<> logger_writer_type;

    namespace detail
    {
        hpx::util::logging::level::type
        get_log_level(std::string const& env, bool allow_always = false)
        {
            try {
                int env_val = boost::lexical_cast<int>(env);
                if (env_val < 0)
                    return hpx::util::logging::level::disable_all;

                switch (env_val) {
                case 0:
                    return allow_always ?
                        hpx::util::logging::level::always :
                        hpx::util::logging::level::disable_all;
                case 1:   return hpx::util::logging::level::fatal;
                case 2:   return hpx::util::logging::level::error;
                case 3:   return hpx::util::logging::level::warning;
                case 4:   return hpx::util::logging::level::info;
                default:
                    break;
                }
                return hpx::util::logging::level::debug;
            }
            catch (boost::bad_lexical_cast const&) {
                return hpx::util::logging::level::disable_all;
            }
        }

        // unescape config entry
        std::string unescape(std::string const &value)
        {
            std::string result;
            std::string::size_type pos = 0;
            std::string::size_type pos1 = value.find_first_of ("\\", 0);
            if (std::string::npos != pos1) {
                do {
                    switch (value[pos1+1]) {
                    case '\\':
                    case '\"':
                    case '?':
                        result = result + value.substr(pos, pos1-pos);
                        pos1 = value.find_first_of ("\\", (pos = pos1+1)+1);
                        break;

                    case 'n':
                        result = result + value.substr(pos, pos1-pos) + "\n";
                        pos1 = value.find_first_of ("\\", pos = pos1+1);
                        ++pos;
                        break;

                    default:
                        result = result + value.substr(pos, pos1-pos+1);
                        pos1 = value.find_first_of ("\\", pos = pos1+1);
                    }

                } while (pos1 != std::string::npos);
                result = result + value.substr(pos);
            }
            else {
            // the string doesn't contain any escaped character sequences
                result = value;
            }
            return result;
        }
    }

    std::string levelname(int level)
    {
        switch (level) {
        case hpx::util::logging::level::enable_all:
            return "     <all>";
        case hpx::util::logging::level::debug:
            return "   <debug>";
        case hpx::util::logging::level::info:
            return "    <info>";
        case hpx::util::logging::level::warning:
            return " <warning>";
        case hpx::util::logging::level::error:
            return "   <error>";
        case hpx::util::logging::level::fatal:
            return "   <fatal>";
        case hpx::util::logging::level::always:
            return "  <always>";
        }

        std::string unknown = std::to_string(level);
        return std::string((std::max)(7-unknown.size(), std::size_t(0)), ' ') +
            "<" + unknown + ">";
    }

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: shepherd
    struct shepherd_thread_id
      : hpx::util::logging::formatter::class_<
            shepherd_thread_id,
            hpx::util::logging::formatter::implement_op_equal::no_context>
    {
        shepherd_thread_id()
        {}

        void operator()(param str) const
        {
            std::stringstream out;
            out << std::hex << std::setw(sizeof(std::size_t)*2)
                << std::setfill('0')
                << hpx::get_worker_thread_num();
            str.prepend_string(out.str());
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: locality prefix
    struct locality_prefix
      : hpx::util::logging::formatter::class_<
            locality_prefix,
            hpx::util::logging::formatter::implement_op_equal::no_context>
    {
        locality_prefix()
        {}

        void operator()(param str) const
        {
            boost::uint32_t locality_id = naming::invalid_locality_id;
            applier::applier* appl = applier::get_applier_ptr();
            if (appl) {
                error_code ec(lightweight);
                locality_id = appl->get_locality_id(ec);
            }

            if (naming::invalid_locality_id != locality_id) {
                std::stringstream out;
                out << std::hex << std::setw(sizeof(boost::uint32_t)*2)
                    << std::setfill('0') << locality_id;
                str.prepend_string(out.str());
            }
            else {
                // called from outside a HPX thread
                str.prepend_string(std::string(sizeof(boost::uint32_t)*2, '-'));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: HPX thread id
    struct thread_id
      : hpx::util::logging::formatter::class_<
            thread_id,
            hpx::util::logging::formatter::implement_op_equal::no_context>
    {
        void operator()(param str) const
        {
            threads::thread_self* self = threads::get_self_ptr();
            if (nullptr != self) {
                // called from inside a HPX thread
                threads::thread_id_type id = threads::get_self_id();
                if (id != threads::invalid_thread_id) {
                    std::stringstream out;
                    out << std::hex << std::setw(sizeof(void*)*2)
                        << std::setfill('0')
                        << reinterpret_cast<std::ptrdiff_t>(id.get());
                    str.prepend_string(out.str());
                    return;
                }
            }

            // called from outside a HPX thread or invalid thread id
            str.prepend_string(std::string(sizeof(void*)*2, '-'));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: HPX thread phase
    struct thread_phase
      : hpx::util::logging::formatter::class_<
            thread_phase,
            hpx::util::logging::formatter::implement_op_equal::no_context>
    {
        void operator()(param str) const
        {
            threads::thread_self* self = threads::get_self_ptr();
            if (nullptr != self) {
                // called from inside a HPX thread
                std::size_t phase = self->get_thread_phase();
                if (0 != phase) {
                    std::stringstream out;
                    out << std::hex << std::setw(sizeof(boost::uint32_t))
                        << std::setfill('0') << self->get_thread_phase();
                    str.prepend_string(out.str());
                    return;
                }
            }

            // called from outside a HPX thread or no phase given
            str.prepend_string(std::string(sizeof(boost::uint32_t), '-'));
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: locality prefix of parent thread
    struct parent_thread_locality
      : hpx::util::logging::formatter::class_<
            parent_thread_locality,
            hpx::util::logging::formatter::implement_op_equal::no_context>
    {
        void operator()(param str) const
        {
            boost::uint32_t parent_locality_id = threads::get_parent_locality_id();
            if (naming::invalid_locality_id != parent_locality_id) {
                // called from inside a HPX thread
                std::stringstream out;
                out << std::hex << std::setw(sizeof(boost::uint32_t)*2)
                    << std::setfill('0') << parent_locality_id;
                str.prepend_string(out.str());
            }
            else {
                // called from outside a HPX thread
                str.prepend_string(std::string(sizeof(boost::uint32_t)*2, '-'));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: HPX parent thread id
    struct parent_thread_id
      : hpx::util::logging::formatter::class_<
            parent_thread_id,
            hpx::util::logging::formatter::implement_op_equal::no_context>
    {
        void operator()(param str) const
        {
            threads::thread_id_repr_type parent_id = threads::get_parent_id();
            if (nullptr != parent_id && threads::invalid_thread_id != parent_id) {
                // called from inside a HPX thread
                std::stringstream out;
                out << std::hex << std::setw(sizeof(void*)*2)
                    << std::setfill('0')
                    << reinterpret_cast<std::ptrdiff_t>(parent_id);
                str.prepend_string(out.str());
            }
            else {
                // called from outside a HPX thread
                str.prepend_string(std::string(sizeof(void*)*2, '-'));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: HPX parent thread phase
    struct parent_thread_phase
      : hpx::util::logging::formatter::class_<
            parent_thread_phase,
            hpx::util::logging::formatter::implement_op_equal::no_context>
    {
        void operator()(param str) const
        {
            std::size_t parent_phase = threads::get_parent_phase();
            if (0 != parent_phase) {
                // called from inside a HPX thread
                std::stringstream out;
                out << std::hex << std::setw(sizeof(boost::uint32_t))
                    << std::setfill('0') << parent_phase;
                str.prepend_string(out.str());
            }
            else {
                // called from outside a HPX thread
                str.prepend_string(std::string(sizeof(boost::uint32_t), '-'));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom formatter: HPX component id of current thread
    struct thread_component_id
      : hpx::util::logging::formatter::class_<
            thread_component_id,
            hpx::util::logging::formatter::implement_op_equal::no_context>
    {
        void operator()(param str) const
        {
            boost::uint64_t component_id = threads::get_self_component_id();
            if (0 != component_id) {
                // called from inside a HPX thread
                std::stringstream out;
                out << std::hex << std::setw(sizeof(boost::uint64_t)*2)
                    << std::setfill('0')
                    << component_id;
                str.prepend_string(out.str());
            }
            else {
                // called from outside a HPX thread
                str.prepend_string(std::string(sizeof(boost::uint64_t)*2, '-'));
            }
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // custom log destination: send generated strings to console
    struct console : hpx::util::logging::destination::is_generic
    {
        console(std::size_t level, logging_destination dest)
          : level_(level), dest_(dest)
        {}

        template<typename MsgType>
        void operator()(MsgType const& msg) const
        {
            components::console_logging(dest_, level_, msg);
        }

        bool operator==(console const& rhs) const
        {
            return dest_ == rhs.dest_;
        }

        std::size_t level_;
        logging_destination dest_;
    };

#if defined(ANDROID) || defined(__ANDROID__)
    // default log destination for Android
    struct android_log : hpx::util::logging::destination::is_generic
    {
        android_log(char const* tag_)
          : tag(tag_)
        {}

        template<typename MsgType>
        void operator()(MsgType const& msg) const
        {
            __android_log_write(ANDROID_LOG_DEBUG, tag.c_str(), msg.c_str());
        }

        bool operator==(android_log const& rhs) const
        {
            return tag == rhs.tag;
        }

        std::string tag;
    };
#endif

    namespace detail
    {
        template <typename Writer>
        void define_formatters(Writer& writer)
        {
            writer.replace_formatter("osthread", shepherd_thread_id());
            writer.replace_formatter("locality", locality_prefix());
            writer.replace_formatter("hpxthread", thread_id());
            writer.replace_formatter("hpxphase", thread_phase());
            writer.replace_formatter("hpxparent", parent_thread_id());
            writer.replace_formatter("hpxparentphase", parent_thread_phase());
            writer.replace_formatter("parentloc", parent_thread_locality());
            writer.replace_formatter("hpxcomponent", thread_component_id());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // this is required in order to use the logging library
    HPX_DEFINE_LOG_FILTER_WITH_ARGS(agas_level, filter_type,
        hpx::util::logging::level::disable_all)
    HPX_DEFINE_LOG(agas_logger, logger_type)

    // initialize logging for AGAS
    void init_agas_log(util::section const& ini, bool isconsole)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.agas")) {
            util::section const* logini = ini.get_section("hpx.logging.agas");
            HPX_ASSERT(nullptr != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            if (!loglevel.empty()) {
                logdest = logini->get_entry("destination", empty);
                logformat = detail::unescape(logini->get_entry("format", empty));
            }
        }

        unsigned lvl = hpx::util::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel);

        if (hpx::util::logging::level::disable_all != lvl)
        {
           logger_writer_type& writer = agas_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())      // ensure minimal defaults
                logdest = isconsole ? "android_log" : "console";
            agas_logger()->writer().add_destination("android_log",
                android_log("hpx.agas"));
#else
            if (logdest.empty())      // ensure minimal defaults
                logdest = isconsole ? "cerr" : "console";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.add_destination("console", console(lvl, destination_agas)); //-V106
            writer.write(logformat, logdest);
            detail::define_formatters(writer);

            agas_logger()->mark_as_initialized();
            agas_level()->set_enabled(lvl);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // this is required in order to use the logging library
    HPX_DEFINE_LOG_FILTER_WITH_ARGS(parcel_level, filter_type,
        hpx::util::logging::level::disable_all)
    HPX_DEFINE_LOG(parcel_logger, logger_type)

    // initialize logging for the parcel transport
    void init_parcel_log(util::section const& ini, bool isconsole)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.parcel")) {
            util::section const* logini = ini.get_section("hpx.logging.parcel");
            HPX_ASSERT(nullptr != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            if (!loglevel.empty()) {
                logdest = logini->get_entry("destination", empty);
                logformat = detail::unescape(logini->get_entry("format", empty));
            }
        }

        unsigned lvl = hpx::util::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel);

        if (hpx::util::logging::level::disable_all != lvl)
        {
           logger_writer_type& writer = parcel_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())      // ensure minimal defaults
                logdest = isconsole ? "android_log" : "console";
            parcel_logger()->writer().add_destination("android_log",
                android_log("hpx.parcel"));
#else
            if (logdest.empty())      // ensure minimal defaults
                logdest = isconsole ? "cerr" : "console";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.add_destination("console",
                console(lvl, destination_parcel)); //-V106
            writer.write(logformat, logdest);
            detail::define_formatters(writer);

            parcel_logger()->mark_as_initialized();
            parcel_level()->set_enabled(lvl);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    // this is required in order to use the logging library for timings
    HPX_DEFINE_LOG_FILTER_WITH_ARGS(timing_level, filter_type,
        hpx::util::logging::level::disable_all)
    HPX_DEFINE_LOG(timing_logger, logger_type)

    // initialize logging for performance measurements
    void init_timing_log(util::section const& ini, bool isconsole)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.timing")) {
            util::section const* logini = ini.get_section("hpx.logging.timing");
            HPX_ASSERT(nullptr != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            if (!loglevel.empty()) {
                logdest = logini->get_entry("destination", empty);
                logformat = detail::unescape(logini->get_entry("format", empty));
            }
        }

        unsigned lvl = hpx::util::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel);

        if (hpx::util::logging::level::disable_all != lvl)
        {
           logger_writer_type& writer = timing_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())      // ensure minimal defaults
                logdest = isconsole ? "android_log" : "console";

            writer.add_destination("android_log",
                android_log("hpx.timing"));
#else
            if (logdest.empty())      // ensure minimal defaults
                logdest = isconsole ? "cerr" : "console";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.add_destination("console", console(lvl, destination_timing)); //-V106
            writer.write(logformat, logdest);
            detail::define_formatters(writer);

            timing_logger()->mark_as_initialized();
            timing_level()->set_enabled(lvl);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    HPX_DEFINE_LOG_FILTER_WITH_ARGS(hpx_level, filter_type,
        hpx::util::logging::level::disable_all)
    HPX_DEFINE_LOG(hpx_logger, logger_type)

    HPX_DEFINE_LOG_FILTER_WITH_ARGS(hpx_error_level, filter_type,
        hpx::util::logging::level::fatal)
    HPX_DEFINE_LOG(hpx_error_logger, logger_type)

    void init_hpx_logs(util::section const& ini, bool isconsole)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging")) {
            util::section const* logini = ini.get_section("hpx.logging");
            HPX_ASSERT(nullptr != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            if (!loglevel.empty()) {
                logdest = logini->get_entry("destination", empty);
                logformat = detail::unescape(logini->get_entry("format", empty));
            }
        }

        unsigned lvl = hpx::util::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel, true);

        logger_writer_type& writer = hpx_logger()->writer();
        logger_writer_type& error_writer = hpx_error_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
        if (logdest.empty())      // ensure minimal defaults
            logdest = isconsole ? "android_log" : "console";

        writer.add_destination("android_log", android_log("hpx"));
        error_writer.add_destination("android_log", android_log("hpx"));
#else
        if (logdest.empty())      // ensure minimal defaults
            logdest = isconsole ? "cerr" : "console";
#endif
        if (logformat.empty())
            logformat = "|\\n";

        if (hpx::util::logging::level::disable_all != lvl) {
            writer.add_destination("console", console(lvl, destination_hpx)); //-V106
            writer.write(logformat, logdest);
            detail::define_formatters(writer);

            hpx_logger()->mark_as_initialized();
            hpx_level()->set_enabled(lvl);

            // errors are logged to the given destination and to cerr
            error_writer.add_destination("console",
                console(lvl, destination_hpx)); //-V106
#if !defined(ANDROID) && !defined(__ANDROID__)
            if (logdest != "cerr")
                error_writer.write(logformat, logdest + " cerr");
#endif
            detail::define_formatters(error_writer);

            hpx_error_logger()->mark_as_initialized();
            hpx_error_level()->set_enabled(lvl);
        }
        else {
            // errors are always logged to cerr
            if (!isconsole) {
                error_writer.add_destination("console",
                    console(lvl, destination_hpx)); //-V106
                error_writer.write(logformat, "console");
            }
            else {
#if defined(ANDROID) || defined(__ANDROID__)
                error_writer.write(logformat, "android_log");
#else
                error_writer.write(logformat, "cerr");
#endif
            }
            detail::define_formatters(error_writer);

            hpx_error_logger()->mark_as_initialized();
            hpx_error_level()->set_enabled(hpx::util::logging::level::fatal);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    HPX_DEFINE_LOG_FILTER_WITH_ARGS(app_level, filter_type,
        hpx::util::logging::level::disable_all)
    HPX_DEFINE_LOG(app_logger, logger_type)

    HPX_DEFINE_LOG_FILTER_WITH_ARGS(app_error_level, filter_type,
        hpx::util::logging::level::fatal)
    HPX_DEFINE_LOG(app_error_logger, logger_type)

    // initialize logging for application
    void init_app_logs(util::section const& ini, bool isconsole)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.application")) {
            util::section const* logini = ini.get_section("hpx.logging.application");
            HPX_ASSERT(nullptr != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            if (!loglevel.empty()) {
                logdest = logini->get_entry("destination", empty);
                logformat = detail::unescape(logini->get_entry("format", empty));
            }
        }

        unsigned lvl = hpx::util::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel);

        if (hpx::util::logging::level::disable_all != lvl) {
            logger_writer_type& writer = app_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())      // ensure minimal defaults
                logdest = isconsole ? "android_log" : "console";
            writer.add_destination("android_log", android_log("hpx.application"));
#else
            if (logdest.empty())      // ensure minimal defaults
                logdest = isconsole ? "cerr" : "console";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.add_destination("console", console(lvl, destination_app)); //-V106
            writer.write(logformat, logdest);
            detail::define_formatters(writer);

            app_logger()->mark_as_initialized();
            app_level()->set_enabled(lvl);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    HPX_DEFINE_LOG_FILTER_WITH_ARGS(debuglog_level, filter_type,
        hpx::util::logging::level::disable_all)
    HPX_DEFINE_LOG(debuglog_logger, logger_type)

    HPX_DEFINE_LOG_FILTER_WITH_ARGS(debuglog_error_level, filter_type,
        hpx::util::logging::level::fatal)
    HPX_DEFINE_LOG(debuglog_error_logger, logger_type)

    // initialize logging for application
    void init_debuglog_logs(util::section const& ini, bool isconsole)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.debuglog")) {
            util::section const* logini = ini.get_section("hpx.logging.debuglog");
            HPX_ASSERT(nullptr != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            if (!loglevel.empty()) {
                logdest = logini->get_entry("destination", empty);
                logformat = detail::unescape(logini->get_entry("format", empty));
            }
        }

        unsigned lvl = hpx::util::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel);

        if (hpx::util::logging::level::disable_all != lvl)
        {
            logger_writer_type& writer = debuglog_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())      // ensure minimal defaults
                logdest = isconsole ? "android_log" : "console";
            writer.add_destination("android_log", android_log("hpx.debuglog"));
#else
            if (logdest.empty())      // ensure minimal defaults
                logdest = isconsole ? "cerr" : "console";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.add_destination("console",
                console(lvl, destination_debuglog)); //-V106
            writer.write(logformat, logdest);
            detail::define_formatters(writer);

            debuglog_logger()->mark_as_initialized();
            debuglog_level()->set_enabled(lvl);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    HPX_DEFINE_LOG_FILTER_WITH_ARGS(agas_console_level, filter_type,
        hpx::util::logging::level::disable_all)
    HPX_DEFINE_LOG(agas_console_logger, logger_type)

    // initialize logging for AGAS
    void init_agas_console_log(util::section const& ini)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.console.agas")) {
            util::section const* logini = ini.get_section("hpx.logging.console.agas");
            HPX_ASSERT(nullptr != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            if (!loglevel.empty()) {
                logdest = logini->get_entry("destination", empty);
                logformat = detail::unescape(logini->get_entry("format", empty));
            }
        }

        unsigned lvl = hpx::util::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel, true);

        if (hpx::util::logging::level::disable_all != lvl)
        {
            logger_writer_type& writer = agas_console_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())      // ensure minimal defaults
                logdest = "android_log";
            writer.add_destination("android_log", android_log("hpx.agas"));
#else
            if (logdest.empty())      // ensure minimal defaults
                logdest = "cerr";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.write(logformat, logdest);

            agas_console_logger()->mark_as_initialized();
            agas_console_level()->set_enabled(lvl);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    HPX_DEFINE_LOG_FILTER_WITH_ARGS(parcel_console_level, filter_type,
        hpx::util::logging::level::disable_all)
    HPX_DEFINE_LOG(parcel_console_logger, logger_type)

    // initialize logging for the parcel transport
    void init_parcel_console_log(util::section const& ini)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.console.parcel")) {
            util::section const* logini =
                ini.get_section("hpx.logging.console.parcel");
            HPX_ASSERT(nullptr != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            if (!loglevel.empty()) {
                logdest = logini->get_entry("destination", empty);
                logformat = detail::unescape(logini->get_entry("format", empty));
            }
        }

        unsigned lvl = hpx::util::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel, true);

        if (hpx::util::logging::level::disable_all != lvl)
        {
            logger_writer_type& writer = parcel_console_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())      // ensure minimal defaults
                logdest = "android_log";
            writer.add_destination("android_log", android_log("hpx.parcel"));
#else
            if (logdest.empty())      // ensure minimal defaults
                logdest = "cerr";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.write(logformat, logdest);

            parcel_console_logger()->mark_as_initialized();
            parcel_console_level()->set_enabled(lvl);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    HPX_DEFINE_LOG_FILTER_WITH_ARGS(timing_console_level, filter_type,
        hpx::util::logging::level::disable_all)
    HPX_DEFINE_LOG(timing_console_logger, logger_type)

    // initialize logging for performance measurements
    void init_timing_console_log(util::section const& ini)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.console.timing")) {
            util::section const* logini = ini.get_section("hpx.logging.console.timing");
            HPX_ASSERT(nullptr != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            if (!loglevel.empty()) {
                logdest = logini->get_entry("destination", empty);
                logformat = detail::unescape(logini->get_entry("format", empty));
            }
        }

        unsigned lvl = hpx::util::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel, true);

        if (hpx::util::logging::level::disable_all != lvl)
        {
            logger_writer_type& writer = timing_console_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())      // ensure minimal defaults
                logdest = "android_log";
            writer.add_destination("android_log", android_log("hpx.timing"));
#else
            if (logdest.empty())      // ensure minimal defaults
                logdest = "cerr";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.write(logformat, logdest);

            timing_console_logger()->mark_as_initialized();
            timing_console_level()->set_enabled(lvl);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    HPX_DEFINE_LOG_FILTER_WITH_ARGS(hpx_console_level, filter_type,
        hpx::util::logging::level::disable_all)
    HPX_DEFINE_LOG(hpx_console_logger, logger_type)

    // initialize logging for HPX runtime
    void init_hpx_console_log(util::section const& ini)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.console")) {
            util::section const* logini = ini.get_section("hpx.logging.console");
            HPX_ASSERT(nullptr != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            if (!loglevel.empty()) {
                logdest = logini->get_entry("destination", empty);
                logformat = detail::unescape(logini->get_entry("format", empty));
            }
        }

        unsigned lvl = hpx::util::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel, true);

        if (hpx::util::logging::level::disable_all != lvl) {
            logger_writer_type& writer = hpx_console_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())      // ensure minimal defaults
                logdest = "android_log";
            writer.add_destination("android_log", android_log("hpx"));
#else
            if (logdest.empty())      // ensure minimal defaults
                logdest = "cerr";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.write(logformat, logdest);

            hpx_console_logger()->mark_as_initialized();
            hpx_console_level()->set_enabled(lvl);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    HPX_DEFINE_LOG_FILTER_WITH_ARGS(app_console_level, filter_type,
        hpx::util::logging::level::disable_all)
    HPX_DEFINE_LOG(app_console_logger, logger_type)

    // initialize logging for applications
    void init_app_console_log(util::section const& ini)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.console.application")) {
            util::section const* logini =
                ini.get_section("hpx.logging.console.application");
            HPX_ASSERT(nullptr != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            if (!loglevel.empty()) {
                logdest = logini->get_entry("destination", empty);
                logformat = detail::unescape(logini->get_entry("format", empty));
            }
        }

        unsigned lvl = hpx::util::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel, true);

        if (hpx::util::logging::level::disable_all != lvl)
        {
            logger_writer_type& writer = app_console_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())      // ensure minimal defaults
                logdest = "android_log";
            writer.add_destination("android_log", android_log("hpx.application"));
#else
            if (logdest.empty())      // ensure minimal defaults
                logdest = "cerr";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.write(logformat, logdest);

            app_console_logger()->mark_as_initialized();
            app_console_level()->set_enabled(lvl);
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    HPX_DEFINE_LOG_FILTER_WITH_ARGS(debuglog_console_level, filter_type,
        hpx::util::logging::level::disable_all)
    HPX_DEFINE_LOG(debuglog_console_logger, logger_type)

    // initialize logging for applications
    void init_debuglog_console_log(util::section const& ini)
    {
        std::string loglevel, logdest, logformat;

        if (ini.has_section("hpx.logging.console.debuglog")) {
            util::section const* logini =
                ini.get_section("hpx.logging.console.debuglog");
            HPX_ASSERT(nullptr != logini);

            std::string empty;
            loglevel = logini->get_entry("level", empty);
            if (!loglevel.empty()) {
                logdest = logini->get_entry("destination", empty);
                logformat = detail::unescape(logini->get_entry("format", empty));
            }
        }

        unsigned lvl = hpx::util::logging::level::disable_all;
        if (!loglevel.empty())
            lvl = detail::get_log_level(loglevel, true);

        if (hpx::util::logging::level::disable_all != lvl)
        {
            logger_writer_type& writer = debuglog_console_logger()->writer();

#if defined(ANDROID) || defined(__ANDROID__)
            if (logdest.empty())      // ensure minimal defaults
                logdest = "android_log";
            writer.add_destination("android_log", android_log("hpx.debuglog"));
#else
            if (logdest.empty())      // ensure minimal defaults
                logdest = "cerr";
#endif
            if (logformat.empty())
                logformat = "|\\n";

            writer.write(logformat, logdest);

            debuglog_console_logger()->mark_as_initialized();
            debuglog_console_level()->set_enabled(lvl);
        }
    }
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // the logging_configuration type will be instantiated exactly once
    struct logging_configuration
    {
        logging_configuration();
        std::vector<std::string> prefill_;
    };

// define the format for the generated time stamps
#define HPX_TIMEFORMAT "$hh:$mm.$ss.$mili"

    logging_configuration::logging_configuration()
    {
        try {
            // add default logging configuration as defaults to the ini data
            // this will be overwritten by related entries in the read hpx.ini
            using namespace boost::assign;
            prefill_ +=
                // general logging
                "[hpx.logging]",
                "level = ${HPX_LOGLEVEL:0}",
                "destination = ${HPX_LOGDESTINATION:console}",
                "format = ${HPX_LOGFORMAT:"
                    "(T%locality%/%hpxthread%.%hpxphase%/%hpxcomponent%) "
                    "P%parentloc%/%hpxparent%.%hpxparentphase% %time%(" HPX_TIMEFORMAT
                    ") [%idx%]|\\n}",

                // general console logging
                "[hpx.logging.console]",
                "level = ${HPX_LOGLEVEL:$[hpx.logging.level]}",
#if defined(ANDROID) || defined(__ANDROID__)
                "destination = ${HPX_CONSOLE_LOGDESTINATION:android_log}",
#else
                "destination = ${HPX_CONSOLE_LOGDESTINATION:\
                    file(hpx.$[system.pid].log)}",
#endif
                "format = ${HPX_CONSOLE_LOGFORMAT:|}",

                // logging related to timing
                "[hpx.logging.timing]",
                "level = ${HPX_TIMING_LOGLEVEL:-1}",
                "destination = ${HPX_TIMING_LOGDESTINATION:console}",
                "format = ${HPX_TIMING_LOGFORMAT:"
                    "(T%locality%/%hpxthread%.%hpxphase%/%hpxcomponent%) "
                    "P%parentloc%/%hpxparent%.%hpxparentphase% %time%(" HPX_TIMEFORMAT
                    ") [%idx%] [TIM] |\\n}",

                // console logging related to timing
                "[hpx.logging.console.timing]",
                "level = ${HPX_TIMING_LOGLEVEL:$[hpx.logging.timing.level]}",
#if defined(ANDROID) || defined(__ANDROID__)
                "destination = ${HPX_CONSOLE_TIMING_LOGDESTINATION:android_log}",
#else
                "destination = \
            ${HPX_CONSOLE_TIMING_LOGDESTINATION:file(hpx.timing.$[system.pid].log)}",
#endif
                "format = ${HPX_CONSOLE_TIMING_LOGFORMAT:|}",

                // logging related to AGAS
                "[hpx.logging.agas]",
                "level = ${HPX_AGAS_LOGLEVEL:-1}",
//                     "destination = ${HPX_AGAS_LOGDESTINATION:console}",
                "destination = \
                ${HPX_AGAS_LOGDESTINATION:file(hpx.agas.$[system.pid].log)}",
                "format = ${HPX_AGAS_LOGFORMAT:"
                    "(T%locality%/%hpxthread%.%hpxphase%/%hpxcomponent%) "
                    "P%parentloc%/%hpxparent%.%hpxparentphase% %time%(" HPX_TIMEFORMAT
                    ") [%idx%][AGAS] |\\n}",

                // console logging related to AGAS
                "[hpx.logging.console.agas]",
                "level = ${HPX_AGAS_LOGLEVEL:$[hpx.logging.agas.level]}",
#if defined(ANDROID) || defined(__ANDROID__)
                "destination = ${HPX_CONSOLE_AGAS_LOGDESTINATION:android_log}",
#else
                "destination = ${HPX_CONSOLE_AGAS_LOGDESTINATION:file\
                    (hpx.agas.$[system.pid].log)}",
#endif
                "format = ${HPX_CONSOLE_AGAS_LOGFORMAT:|}",

                // logging related to the parcel transport
                "[hpx.logging.parcel]",
                "level = ${HPX_PARCEL_LOGLEVEL:-1}",
                "destination = ${HPX_PARCEL_LOGDESTINATION:file\
                (hpx.parcel.$[system.pid].log)}",
                "format = ${HPX_PARCEL_LOGFORMAT:"
                    "(T%locality%/%hpxthread%.%hpxphase%/%hpxcomponent%) "
                    "P%parentloc%/%hpxparent%.%hpxparentphase% %time%(" HPX_TIMEFORMAT
                    ") [%idx%][  PT] |\\n}",

                // console logging related to the parcel transport
                "[hpx.logging.console.parcel]",
                "level = ${HPX_PARCEL_LOGLEVEL:$[hpx.logging.parcel.level]}",
#if defined(ANDROID) || defined(__ANDROID__)
                "destination = ${HPX_CONSOLE_PARCEL_LOGDESTINATION:android_log}",
#else
                "destination = ${HPX_CONSOLE_PARCEL_LOGDESTINATION:file\
                    (hpx.parcel.$[system.pid].log)}",
#endif
                "format = ${HPX_CONSOLE_PARCEL_LOGFORMAT:|}",

                // logging related to applications
                "[hpx.logging.application]",
                "level = ${HPX_APP_LOGLEVEL:-1}",
                "destination = ${HPX_APP_LOGDESTINATION:console}",
                "format = ${HPX_APP_LOGFORMAT:"
                    "(T%locality%/%hpxthread%.%hpxphase%/%hpxcomponent%) "
                    "P%parentloc%/%hpxparent%.%hpxparentphase% %time%(" HPX_TIMEFORMAT
                    ") [%idx%] [APP] |\\n}",

                // console logging related to applications
                "[hpx.logging.console.application]",
                "level = ${HPX_APP_LOGLEVEL:$[hpx.logging.application.level]}",
#if defined(ANDROID) || defined(__ANDROID__)
                "destination = ${HPX_CONSOLE_APP_LOGDESTINATION:android_log}",
#else
                "destination = ${HPX_CONSOLE_APP_LOGDESTINATION:file\
                    (hpx.application.$[system.pid].log)}",
#endif
                "format = ${HPX_CONSOLE_APP_LOGFORMAT:|}",

                // logging of debug channel
                "[hpx.logging.debuglog]",
                "level = ${HPX_DEB_LOGLEVEL:-1}",
                "destination = ${HPX_DEB_LOGDESTINATION:console}",
                "format = ${HPX_DEB_LOGFORMAT:"
                    "(T%locality%/%hpxthread%.%hpxphase%/%hpxcomponent%) "
                    "P%parentloc%/%hpxparent%.%hpxparentphase% %time%(" HPX_TIMEFORMAT
                    ") [%idx%] [DEB] |\\n}",

                "[hpx.logging.console.debuglog]",
                "level = ${HPX_DEB_LOGLEVEL:$[hpx.logging.debuglog.level]}",
#if defined(ANDROID) || defined(__ANDROID__)
                "destination = ${HPX_CONSOLE_DEB_LOGDESTINATION:android_log}",
#else
                "destination = ${HPX_CONSOLE_DEB_LOGDESTINATION:file\
                    (hpx.debuglog.$[system.pid].log)}",
#endif
                "format = ${HPX_CONSOLE_DEB_LOGFORMAT:|}"
            ;
        }
        catch (std::exception const&) {
            // just in case something goes wrong
            std::cerr << "caught std::exception during initialization"
                      << std::endl;
        }
    }

#undef HPX_TIMEFORMAT

    struct init_logging_tag {};
    std::vector<std::string> const& get_logging_data()
    {
        static_<logging_configuration, init_logging_tag> init;
        return init.get().prefill_;
    }

    ///////////////////////////////////////////////////////////////////////////
    init_logging::init_logging(runtime_configuration& ini, bool isconsole,
        naming::resolver_client& agas_client)
    {
        // initialize normal logs
        init_agas_log(ini, isconsole);
        init_parcel_log(ini, isconsole);
        init_timing_log(ini, isconsole);
        init_hpx_logs(ini, isconsole);
        init_app_logs(ini, isconsole);
        init_debuglog_logs(ini, isconsole);

        // initialize console logs
        init_agas_console_log(ini);
        init_parcel_console_log(ini);
        init_timing_console_log(ini);
        init_hpx_console_log(ini);
        init_app_console_log(ini);
        init_debuglog_console_log(ini);
    }
}}}

#else  // HPX_HAVE_LOGGING

#include <hpx/util/runtime_configuration.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/init_logging.hpp>

#include <iostream>
#include <string>
#include <vector>

namespace hpx { namespace util { namespace detail
{
    dummy_log_impl dummy_log = dummy_log_impl();

    std::vector<std::string> get_logging_data()
    {
        static std::vector<std::string> dummy_data;
        return dummy_data;
    }

    init_logging::init_logging(runtime_configuration& ini, bool,
        naming::resolver_client&)
    {
        // warn if logging is requested

        if (ini.get_entry("hpx.logging.level", "0") != "0" ||
            ini.get_entry("hpx.logging.timing.level", "0") != "0" ||
            ini.get_entry("hpx.logging.agas.level", "0") != "0" ||
            ini.get_entry("hpx.logging.debuglog.level", "0") != "0" ||
            ini.get_entry("hpx.logging.application.level", "0") != "0")
        {
            std::cerr << "hpx::init_logging: warning: logging is requested even "
                         "while it was disabled at compile time. If you "
                         "need logging to be functional, please reconfigure and "
                         "rebuild HPX with HPX_WITH_LOGGING set to ON."
                      << std::endl;
        }
    }
}}}

#endif // HPX_HAVE_LOGGING
