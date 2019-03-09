// named_spacer.hpp

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details


#ifndef JT28092007_named_spacer_HPP_DEFINED
#define JT28092007_named_spacer_HPP_DEFINED

#include <hpx/util/logging/detail/fwd.hpp>
#include <hpx/util/logging/detail/manipulator.hpp>
#include <hpx/util/logging/format/array.hpp>                       // array
#include <hpx/util/logging/format/formatter/convert_format.hpp>    // do_convert_format

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace hpx { namespace util { namespace logging { namespace formatter {

namespace detail {

    inline std::string unescape(std::string escaped) {
        typedef std::size_t size_type;
        size_type idx_start = 0;
        while ( true) {
            size_type found = escaped.find( "%%", idx_start );
            if ( found != std::string::npos) {
                escaped.erase( escaped.begin() +
                    static_cast<std::ptrdiff_t>(found));
                ++idx_start;
            }
            else
                break;
        }
        return escaped;
    }

    struct base_named_spacer_context {
        typedef base format_base_type;
        typedef ::hpx::util::logging::array::shared_ptr_holder<format_base_type >
            array;

        struct write_step {
            write_step(const std::string & prefix_, format_base_type * fmt_)
                : prefix(prefix_), fmt(fmt_) {}
            std::string prefix;
            // could be null - in case formatter not found by name, or it's the last step
            format_base_type * fmt;
        };

        struct write_info {
            array formatters;
            typedef std::map<std::string, format_base_type* > coll;
            coll name_to_formatter;

            std::string format_string;

            // how we write
            typedef std::vector<write_step> write_step_array;
            write_step_array write_steps;
        };
        write_info m_info;

        template<class formatter, class converter>
        void add(const std::string & name, formatter fmt) {
            // care about if generic or not
            typedef hpx::util::logging::manipulator::is_generic is_generic;
            add_impl<converter, formatter>(
                name, fmt, std::is_base_of<is_generic,formatter>() );
            compute_write_steps();
        }

        void del(const std::string & name) {
            {
            format_base_type * p = m_info.name_to_formatter[name];
            m_info.name_to_formatter.erase(name);
            m_info.formatters.del(p);
            }
            compute_write_steps();
        }

        void configure(const std::string & name, const std::string & configure_str) {
            format_base_type * p = m_info.name_to_formatter[name];
            if ( p)
                p->configure(configure_str);
        }

        void format_string(const std::string & str) {
            {
              m_info.format_string = str;
            }
            compute_write_steps();
        }

    protected:
        template<class converter>
        void write_with_convert(msg_type & msg,
            ::hpx::util::logging::formatter::do_convert_format::prepend*) const {
            // prepend
            typedef typename write_info::write_step_array array_;
            for ( typename array_::const_reverse_iterator b =
                m_info.write_steps.rbegin(),
                e = m_info.write_steps.rend(); b != e; ++b) {
                if ( b->fmt)
                    (*(b->fmt))(msg);
                converter::write( b->prefix, msg);
            }
        }
        template<class converter>
        void write_with_convert(msg_type & msg, ...) const {
            // append
            typedef typename write_info::write_step_array array_;
            for ( typename array_::const_iterator b = m_info.write_steps.begin(),
                e = m_info.write_steps.end(); b != e; ++b) {
                converter::write( b->prefix, msg);
                if ( b->fmt)
                    (*(b->fmt))(msg);
            }
        }

        // recomputes the write steps - note taht this takes place after each operation
        // for instance, the user might have first set the string and
        // later added the formatters
        void HPX_EXPORT compute_write_steps();

    private:
        // non-generic
        template<class convert, class formatter>
        void add_impl(const std::string & name,
            formatter fmt, const std::false_type& ) {
            format_base_type * p = m_info.formatters.append(fmt);
            m_info.name_to_formatter[name] = p;
        }
        // generic manipulator
        template<class convert, class formatter>
        void add_impl(const std::string & name,
            formatter fmt, const std::true_type& ) {
            typedef hpx::util::logging::manipulator::detail::generic_holder<formatter,
                format_base_type> holder;

            typedef typename formatter::convert_type formatter_convert_type;
            // they must share the same type of conversion
            // - otherwise when trying to prepend we could end up appending or vice versa
            static_assert( (std::is_same<formatter_convert_type, convert>::value),
                "std::is_same<formatter_convert_type, convert_type>::value");

            add_impl<convert>( name, holder(fmt), std::false_type() );
        }


    };

    template<class convert>
    struct named_spacer_context : base_named_spacer_context {
        typedef convert convert_type;

        template<class formatter> void add(const std::string & name, formatter fmt) {
            base_named_spacer_context::add<formatter, convert>(name, fmt );
        }

        void write(msg_type & msg) const {
            base_named_spacer_context::write_with_convert<convert>( msg, nullptr );
        }
    };
}

/**
@brief Allows you to contain multiple formatters,
and specify a %spacer between them. You have a %spacer string, and within it,
you can escape your contained formatters.

@code
#include <hpx/util/logging/format/formatter/named_spacer.hpp>
@endcode

This allows you:
- to hold multiple formatters
- each formatter is given a name, when being added
- you have a %spacer string, which contains what is to be prepended or
appended to the string (by default, prepended)
- a formatter is escaped with @c '\%' chars, like this @c "%name%"
- if you want to write the @c '\%', just double it,
like this: <tt>"this %% gets written"</tt>

Example:

@code
#define L_ HPX_LOG_USE_LOG_IF_FILTER(g_l(), g_log_filter()->is_enabled() )

g_l()->writer().add_formatter( formatter::named_spacer("[%index%] %time% (T%thread%) ")
        .add( "index", formatter::idx())
        .add( "thread", formatter::thread_id())
        .add( "time", formatter::time("$mm")) );
@endcode

Assuming you'd use the above in code
@code
int i = 1;
L_ << "this is so cool " << i++;
L_ << "this is so cool again " << i++;
@endcode

You could have an output like this:

@code
[1] 53 (T3536) this is so cool 1
[2] 54 (T3536) this is so cool again 2
@endcode

*/
template<class convert>
struct named_spacer_t : is_generic,
    non_const_context< detail::named_spacer_context<convert> > {

    typedef non_const_context< detail::named_spacer_context<convert> > context_base;

    named_spacer_t(const std::string & str = std::string() ) {
        if ( !str.empty() )
            context_base::context().format_string(str);
    }

    named_spacer_t & string(const std::string & str) {
        context_base::context().format_string(str);
        return *this;
    }

    template<class formatter> named_spacer_t & add(const std::string & name,
        formatter fmt) {
        context_base::context().add(name, fmt);
        return *this;
    }

    void del(const std::string & name) {
        context_base::context().del(name);
    }

    void configure_inner(const std::string & name, const std::string & configure_str) {
        context_base::context().configure(name, configure_str);
    }

    void operator()(msg_type & msg) const {
        context_base::context().write(msg);
    }

    bool operator==(const named_spacer_t & other) const {
        return &( context_base::context() ) == &( other.context_base::context() );
    }

};

}}}}

#endif
