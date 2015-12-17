// manipulator.hpp

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


#ifndef JT28092007_manipulator_HPP_DEFINED
#define JT28092007_manipulator_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
// 'class1' : inherits 'class2::member' via dominance
#pragma warning ( disable : 4250)
#endif

#include <hpx/util/logging/detail/fwd.hpp>
#include <hpx/util/logging/detail/format_msg_type.hpp>

#include <hpx/util/logging/format/op_equal.hpp>
#include <hpx/util/logging/detail/forward_constructor.hpp>
// HPX_LOGGING_FORWARD_CONSTRUCTOR_WITH_NEW
#include <boost/shared_ptr.hpp>


namespace hpx { namespace util { namespace logging {


/**
@brief Manipulators = Formatters and/or destinations.


- @ref manipulator_common
- @ref manipulator_base_class
- @ref manipulator_default_base_class
- @ref manipulator_generic
- @ref manipulator_create
- @ref manipulator_share_data
- @ref manipulator_manipulate


\n\n\n
@section manipulator_common Common base class

All formatters need to derive from a <b>common %base class</b>.
Same goes for destinations.

Remember:
- formatter - allows formatting the message before writing it
(like, prepending extra information - an index, the time, thread id, etc)
- destination - is a place where the message is to be written to
(like, the console, a file, a socket, etc)

In your @ref hpx::util::logging::writer::format_write "format_write" object,
you can have several formatters and destinations.
Note that each formatter class and each destination class is a @c %manipulator.

Each formatter and destination classes implement
<tt>operator()(arg_type msg);</tt>, which
processes the message:
- for a formatter, this formats the msg
  (like, prepends time to it, appends an enter, etc)
- for a destination, this writes the message to a destination (like,
to console, a file, etc)





\n\n\n
@section manipulator_base_class Specifying the base class

You can use a typedef - one for the formatters, and one for the destinations:

@code
// ptr_type - optional ; usualy  you don't need to worry about this
typedef formatter::base<  arg_type [,ptr_type] > formatter_base;
typedef destination::base< arg_type [,ptr_type] > destination_base;
@endcode


The @c arg_type is the argument you receive in your <tt>operator()</tt>,
to process the message. It can be as simple as this:

@code
// formatter - needs to modify the message
typedef formatter::base< std::string&> formatter_base;

// destination - needs to write the message - usually,
it doesn't need to modify the message
typedef destination::base<const std::string &> destination_base;
@endcode

Or, you can use a @ref customize_manipulator "custom string class", or,
even an @ref customize_optimize "optimization string class".
So, it's not uncommon to do something like this:

@code
typedef optimize::cache_string_one_str<> cache_string;

// formatter - needs to modify the message - use an optimizer while formatting
typedef formatter::base< cache_string&> formatter_base;

// destination - needs to write the message - which has been converted to string
typedef destination::base<const std::string &> destination_base;
@endcode




\n\n\n
@section manipulator_default_base_class Default base classes

As shown above, you can do your own typedefs. But there's an easier way,
to specify the default base classes:
use the default formatter %base class and the default destination %base class.

They are: <tt>formatter::base<> </tt> and <tt>destination::base<> </tt>.

The default formatter %base class is computed based on your usage of the
@ref HPX_LOG_FORMAT_MSG macro:
- if you haven't used it, it's <tt>std::(w)string & </tt>
- if you've used it, it's the type you specified there; see below

@code
HPX_LOG_FORMAT_MSG( optimize::cache_string_several_str<> )
@endcode

In the above case
@code
formatter::base<> = formatter::base< optimize::cache_string_several_str<>& >
@endcode



The default destination %base class is computed based on your usage of the
@ref HPX_LOG_DESTINATION_MSG macro:
- if you haven't used it, it's <tt>const std::(w)string & </tt>
- if you've used it, it's the type you specified there; see below

@code
HPX_LOG_DESTINATION_MSG( my_cool_string )
@endcode

In the above case
@code
destination::base<> = destination::base< const my_cool_string & >
@endcode





\n\n\n
@section manipulator_generic Using manipulators that come with the library

Now, you will define your @ref logger "logger(s)",
to use the @ref hpx::util::logging::writer::format_write "format_write" class:

@code
HPX_DECLARE_LOG(g_l, logger_format_write< formatter_base,destination_base> > );
@endcode

After this, you'll add formatter and/or destination classes to your logger(s):

@code
// add formatters : [idx] [time] message [enter]
g_l()->writer().add_formatter( formatter::idx() );
g_l()->writer().add_formatter( formatter::time() );
g_l()->writer().add_formatter( formatter::append_newline() );

// write to cout and file
g_l()->writer().add_destination( destination::cout() );
g_l()->writer().add_destination( destination::file("out.txt") );
@endcode

In the above case, if you were to write:

@code
#define L_ ... // defining the logger

int i = 1;
L_ << "this is so cool" << i++;
@endcode

a message similar to this would appear on both the console, and the file:

@code
[1] 12:57 this is so cool 1 <enter>
@endcode


You can use the formatter and/or destination classes that come with the library:
- formatters: in the formatter namespace. Here are a few examples:
  - formatter::idx - prepends an index
  - formatter::append_newline - appends an enter after the message
  - formatter::append_newline_if_needed - appends an enter after the message,
  if not already there
  - formatter::time - prepends the time
  - formatter::thread_id - prepends the current thread id
- destinations: in the destination namespace
  - destination::cout - writes to console
  - destination::stream - writes to a stream
  - destination::file - writes to file
  - destination::rolling_file - writes to a rolling file
  - destination::shared_memory - writes into shared memory
  (using @c hpx::util::shmem::named_shared_object)

Or, you can create your own formatter and/or destination class. See below:



\n\n\n
@section manipulator_create Creating your own formatter and/or destination class(es)

To create your formatter class, you need to derive from
@ref class_ "formatter::class_". You will need to implement
<tt>operator()(arg_type)</tt> <br>
(@c arg_type is the argument from your
@ref manipulator_base_class "formatter base class")

@code
// milliseconds since start of the program
struct ms_since_start : formatter::class_<ms_since_start,
formatter::implement_op_equal::no_context> {
    time_t m_start;
    ms_since_start : m_start( time(0) ) {}

    // param = std::string&
    // (in other words, it's the arg_type from your formatter base class)
    void operator()(param msg) const {
        std::ostringstream out;
        time_t now = time(0);
        out << "[" << (now-start) << "] ";
        msg = out.str() + msg;
    }
};
@endcode

To create your destination class, you need to derive from
@ref class_ "destination::class_". You will need to implement
<tt>operator()(arg_type)</tt> <br>
(@c arg_type is the argument from your @ref manipulator_base_class
"destination base class")

@code
struct to_hwnd
: destination::class_<to_hwnd, destination::implement_op_equal::has_context> {
    HWND h;
    to_hwnd(HWND h) : h(h) {}

    bool operator==(const to_hwnd& other) { return h == other.h; }

    // param = const std::string&
    // (in other words, it's the arg_type from your destination base class)
    void operator()(param msg) const {
        ::SetWindowText(h, msg.c_str());
    }
};
@endcode



\n\n\n
@section manipulator_share_data Sharing data for manipulator classes

When you implement your own %manipulator (%formatter or %destination) class,
you must make sure that
it behaves like an STL function: <b>it needs to contain data as constant.</b>

As long as data is constant, it's all ok - that is, no matter what functions get called,
all the data in the formatter/destination
must remain constant. We need constant functors - just like in STL - because internally,
we copy formatters/destinations: that is, we keep
several copies of a certain object - they all need to be syncronized.
In case the objects' data is constant, that's no problem.

In case the data needs to be changed - it needs to be shared.
Several copies of the same instance must point to the same data.
I've already provided a class you can derive from,
when this is the case: the non_const_context class.

@code
struct my_file : destination::class_<my_file,destination_base,op_equal_has_context>,
destination::non_const_context<std::ofstream> {
    std::string m_filename;
    bool operator==(const my_file & other) { return m_filename == other.m_filename; }

    write_to_file(const std::string & filename)
    : m_filename(filename), non_const_context_base(filename.c_str()) {}
    void operator()(param msg) const {
        context() << msg << std::endl ;
    }
};
@endcode


\n\n\n
@section manipulator_manipulate Modifying a manipulator's state

When it comes to keeping its state, a manipulator (formatter or destination) instance,
has 2 possibilities:
-# either all its member data is constant - in which case you can't manipulate
it (you can't modify it), OR
-# it has non const information, which can change, and thus, some can be manipulated

In the former case, all the member functions the manipulator exposes
are <tt>const</tt>ant.

In the latter case,
- your manipulator class can have member functions that can change its state
(non-const member functions).
- your manipulator class @b must use the non_const_context class to hold
all its non-const state

What this guarantees is @ref non_const_pointer_semantics "pointer-like semantics".

Assume that you have your logger that uses formatters and destinations.
You've added a manipulator to your logger,
and at a later time, you want to modify it (the manipulator, that is).
To achieve this, you'll create a copy, and modify that one (this will work
because of the @ref non_const_pointer_semantics "pointer-like semantics"):

Example 1: reusing the same %destination for 2 logs

@code
destination::file out("out.txt");
g_l_dbg()->writer().add_destination(out);
g_l_app()->writer().add_destination(out);
@endcode

\n
Example 2: allow resetting/clearing a destination's stream

@code
// allow resetting a destination's stream
destination::stream g_out(std::cout);
g_l()->writer().add_destintination(g_out);

// assuming this uses g_l(), this will output to std::cout
L_ << "hello world";

g_out.stream(&std::cerr);
// assuming this uses g_l(), this will output to std::cerr
L_ << "hello world 2";

g_out.clear();
// assuming this uses g_l(), this will not output anything
L_ << "hello world 3";

@endcode




\n\n\n
@section manipulator_use_it Using loggers in code

Now that you've @ref manipulator_generic "added" formatters and/or destinations,
you'll @ref defining_logger_macros "define the macros through which you'll do logging",
and then do logging in your code:

@code
// macros through which you'll do logging
#define LDBG_ HPX_LOG_USE_LOG_IF_LEVEL(g_l(), g_log_level(), debug )
#define LERR_ HPX_LOG_USE_LOG_IF_LEVEL(g_l(), g_log_level(), error )
#define LAPP_ HPX_LOG_USE_LOG_IF_LEVEL(g_l(), g_log_level(), info )

// doing logging in code
int i = 1;
LDBG_ << "this is so cool " << i++;
LERR_ << "first error " << i++;

std::string hello = "hello", world = "world";
LAPP_ << hello << ", " << world;

g_log_level()->set_enabled(level::error);
LDBG_ << "this will not be written anywhere";
LAPP_ << "this won't be written anywhere either";
LERR_ << "second error " << i++;

g_log_level()->set_enabled(level::info);
LAPP_ << "good to be back ;) " << i++;
LERR_ << "third error " << i++;

@endcode

*/
namespace manipulator {

/**
    @brief What to use as base class, for your manipulator classes

    When using formatters and destinations, formatters must share a %base class,
    and destinations must share a %base class - see manipulator namespace.

    @note
    Don't use directly. Use formatter::base<> or destination::base<> instead.
*/
template<
        class raw_param_type,
        class param_type,
        class ptr_type_ = default_ >
    struct base : hpx::util::logging::op_equal::same_type_op_equal_base {

    typedef base<raw_param_type, param_type, ptr_type_> self_type;

    typedef typename use_default<ptr_type_, self_type*>::type ptr_type;

    // used as msg_type in format_and_write classes
    typedef raw_param_type raw_param;
    typedef param_type param;

    virtual void operator()(param val) const = 0;

    /** @brief Override this if you want to allow configuration through scripting

    That is, this allows configuration of your manipulator (formatter/destination)
    at run-time.
    */
    virtual void configure(const hold_string_type& ) {}

protected:
    // signify that we're only a base class - not to be used directly
    base() {}
    virtual ~base() {}
};




/**
    @brief When you implement your manipulator class,
    how is operator== to be implemented?
*/
struct implement_op_equal {
    enum type {
        /// manipulator has no context
        /// - that is, any two values of this type are considered equal
        /// (operator== will automatically return true)
        no_context,
        /// manipulator has context - that is, you <b>have to</b> implement operator==
        /// in your manipulator class
        has_context
    };
};

namespace detail {
    template<implement_op_equal::type> struct op_equal_base {
        bool operator==(const op_equal_base& ) const { return true; }
    };

    template<> struct op_equal_base<implement_op_equal::has_context> {};
}

/**
    @brief Use this when implementing your own formatter or destination class.
    Don't use this directly. Use formatter::class_ or destination::class_
*/
template<class type, implement_op_equal::type op_e, class base_type> struct class_
        : base_type,
          detail::op_equal_base<op_e>,
          hpx::util::logging::op_equal::same_type_op_equal<type> {

    /** @brief Override this if you want to allow configuration through scripting

    That is, this allows configuration of your manipulator
    (formatter/destination) at run-time.
    */
    virtual void configure(const hold_string_type& ) {}

};



/** @brief In case your manipulator (formatter or destination)
needs to hold non-const context information, it can to derive from this.
This automatically creates a shared pointer to the context information.

Also, it provides the following operations:

@c context(), which returns a <tt>context_type &</tt> reference

Example:

@code
struct write_to_file : destination_base,
destination::non_const_context<std::ofstream> {
write_to_file(const char* filename) : non_const_context_base(filename) {}
void operator()(param msg) const {
    context() << msg ;
}
};
@endcode

@section non_const_pointer_semantics non_const_context - Pointer-like semantics

Using non_const_context guarantees @em pointer-like semantics:
if you copy-construct or copy-assign a value, both values will point to the same context:

@code
write_to_file a, b = a;
a.file_name("t1.txt");
// a == b  (a's state == b's state)

write_to_file c, d;
c.file_name("t2.txt");
// c != d  (c's state != d's state)

d = c;
c.file_name("t3.txt");
// c == d  (c's state == d's state)
@endcode


@remarks
In case your manipulator has constant data, you don't need this
*/
template<class context_type> struct non_const_context { //-V690

    // this can be used in the parent class, to forward data from its constructor
    typedef non_const_context<context_type> non_const_context_base;
private:
    typedef non_const_context<context_type> self_type;
    typedef boost::shared_ptr<context_type> ptr_type;

protected:
    non_const_context(const non_const_context& other) : m_context(other.m_context) {}

    HPX_LOGGING_FORWARD_CONSTRUCTOR_WITH_NEW(non_const_context,m_context,context_type)

    context_type & context() const    { return *(m_context.get()); }

private:
    mutable ptr_type m_context;
};



/**
@brief Represents a generic manipulator (formatter or destination)

A generic manipulator is one that does not derive from any formatter_base
or destination_base class (@ref manipulator_base_class).

Libraries, such as this one, can provide generic manipulators,
and they can't rely on
any @ref manipulator_base_class "base class" - since it's you,
the user, who can choose which is the base class.

A generic manipulator has no way of knowing the type of the
@em msg you pass on operator().
Thus, usually generic manipulators have a templated operator=,
and do the best to convert what's in, to what they need.

Example:
@code
template<class convert_dest = do_convert_destination > struct cout {
    template<class msg_type> void operator()(const msg_type & msg) const {
        convert_dest::write(msg, std::cout);
    }
};
@endcode

As long as exists a conversion function from your @c msg_type to what
the manipulator needs, it all works.
Thus, no matter what your %formatter @ref manipulator_base_class "base class" or
%destination @ref manipulator_base_class "base class"
is, the code will still work. You can add your %formatter/ %destination classes,
and the generic %formatter/ %destination classes

@code
typedef ... formatter_base;
logger< format_write<...> > g_l();

struct my_cool_formatter : formatter_base { ... };

// adding formatter class from the Logging lib
g_l().add_formatter( formatter::thread_id() );

// adding formatter class defined by you
g_l().add_formatter( my_cool_formatter() );
@endcode

@sa hpx::util::logging::destination::convert, hpx::util::logging::formatter::convert
*/
struct is_generic {
    virtual ~is_generic() {}

    /** @brief Override this if you want to allow configuration through scripting

    That is, this allows configuration of your manipulator
    (formatter/destination) at run-time.
    */
    virtual void configure(const hold_string_type& ) {}
};

namespace detail {

    // holds the generic manipulator, and forwards to it
    template<class generic_type, class manipulator_base> struct generic_holder
            : class_<
                    generic_holder<generic_type,manipulator_base>,
                    implement_op_equal::has_context,
                    manipulator_base > {
        typedef typename manipulator_base::param param;

        generic_type m_val;
        generic_holder(const generic_type & val) : m_val(val) {}

        bool operator==(const generic_holder& other) const {
            return m_val == other.m_val;
        }

        virtual void operator()(param val) const {
            m_val.operator()(val);
        }

        virtual void configure(const hold_string_type& str) {
            m_val.configure(str);
        }
    };
}













} // namespace manipulator

/**
@brief Formatter is a manipulator.
It allows you to format the message before writing it to the destination(s)

Examples of formatters are : @ref formatter::time_t "prepend the time",
@ref formatter::high_precision_time_t "prepend high-precision time",
@ref formatter::idx_t "prepend the index of the message", etc.


See:
- @ref manipulator "The manipulator namespace"
- @ref manipulator_manipulate "Modifying a formatter's state"
- @ref manipulator::non_const_context "formatter::non_const_context"

*/
namespace formatter {
    namespace detail {
        template<class arg, class ptr, class dummy = override >
        struct format_base_finder {
            typedef typename use_default<arg,
                typename hpx::util::logging::formatter::msg_type<dummy>::type >
                ::type arg_type;
            typedef hpx::util::logging::manipulator::base< arg_type, arg_type &, ptr>
                type;
        };
    }

    /**
    @brief What to use as base class, for your formatter classes

    When using formatters and destinations, formatters must share a %base class,
    and destinations must share a %base class - see manipulator namespace.
    */
    template<
        // note: I'm counting on these defaults, in format_find_writer class
        class arg_type = default_ ,
        class ptr_type_ = default_ >
    struct base : detail::format_base_finder<arg_type,ptr_type_>::type {
    };

    /**
        @sa hpx::util::logging::manipulator::implement_op_equal
    */
    typedef hpx::util::logging::manipulator::implement_op_equal implement_op_equal;

    /**
        @brief Use this when implementing your own formatter class

        @param type Your own class name
        @param op_e How will you @ref
        hpx::util::logging::manipulator::implement_op_equal "implement operator=="

        @param base_type (optional) The formatter base class.
        Unless you've specified your own formatter class,
        you'll be happy with the default
    */
    template<class type, implement_op_equal::type op_e, class base_type =
          base<> > struct class_
        : hpx::util::logging::manipulator::class_<type, op_e, base_type> {};


    using hpx::util::logging::manipulator::non_const_context;

    /**
        @sa hpx::util::logging::manipulator::is_generic
    */
    typedef hpx::util::logging::manipulator::is_generic is_generic;

}

/**
@brief Destination is a manipulator. It contains a place where the message,
after being formatted, is to be written to.

Some viable destinations are : @ref destination::cout_t "the console",
@ref destination::file_t "a file", a socket, etc.

See:
- @ref manipulator "The manipulator namespace"
- @ref manipulator_manipulate "Modifying a destination's state"
- @ref manipulator::non_const_context "formatter::non_const_context"


*/
namespace destination {
    namespace detail {
        template<class arg, class ptr, class dummy = override >
        struct destination_base_finder {
            typedef typename use_default<arg,
                typename hpx::util::logging::destination::msg_type<dummy>::type >
                ::type arg_type;
            typedef hpx::util::logging::manipulator::base< arg_type,
                const arg_type &, ptr> type;
        };
    }

    /**
    @brief What to use as base class, for your destination classes

    When using formatters and destinations, formatters must share a %base class,
    and destinations must share a %base class - see manipulator namespace.
    */
    template<
        // note: I'm counting on these defaults, in format_find_writer class
        class arg_type = default_ ,
        class ptr_type_ = default_ >
    struct base : detail::destination_base_finder<arg_type,ptr_type_>::type {
    };

    using hpx::util::logging::manipulator::non_const_context;

    /**
        @sa hpx::util::logging::manipulator::implement_op_equal
    */
    typedef hpx::util::logging::manipulator::implement_op_equal implement_op_equal;

    /**
        @brief Use this when implementing your own destination class

        @param type Your own class name
        @param op_e How will you @ref
        hpx::util::logging::manipulator::implement_op_equal "implement operator=="

        @param base_type (optional)
        The destination base class. Unless you've specified your own destination class,
        you'll be happy with the default
    */
    template<class type, implement_op_equal::type op_e, class base_type =
          base<> > struct class_
        : hpx::util::logging::manipulator::class_<type, op_e, base_type> {};

    /**
        @sa hpx::util::logging::manipulator::is_generic
    */
    typedef hpx::util::logging::manipulator::is_generic is_generic;

}

}}}

#endif

