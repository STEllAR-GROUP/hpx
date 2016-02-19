// op_equal.hpp

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

#ifndef JT28092007_op_equal_HPP_DEFINED
#define JT28092007_op_equal_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#include <hpx/util/logging/detail/fwd.hpp>
#include <vector>
#include <typeinfo>

namespace hpx { namespace util { namespace logging {

/**
    @brief Implements operator== for manipulators

*/
namespace op_equal {



    struct same_type_op_equal_top {
        virtual bool equals(const same_type_op_equal_top &) const = 0;
    protected:
        same_type_op_equal_top() {}
        virtual ~same_type_op_equal_top() {}
        same_type_op_equal_top(const same_type_op_equal_top&) {}
    };
    inline bool operator ==(const same_type_op_equal_top& a,
        const same_type_op_equal_top&b) { return a.equals(b); }

    /**
        @brief Base class when you want to implement operator==
        that will compare based on type and member operator==

        @sa same_type_op_equal
    */
    struct same_type_op_equal_base : virtual same_type_op_equal_top {};

    struct always_equal {
        bool operator==(const always_equal& ) const { return true; }
    };

    /**
        @brief Implements operator==, which compares two objects.
        If they have the same type, it will compare them using the type's member
        operator==.

        The only constraint is that operator== must be a *member* function
    */
    template<class type> struct same_type_op_equal : same_type_op_equal_base {

        virtual bool equals(const same_type_op_equal_top & other) const {
            if ( typeid(*this) != typeid(other))
                return false;
            const type & real_other = dynamic_cast<const type&>(other);

            // this forces 'type' to implement operator==
            return (dynamic_cast<const type&>(*this)).operator ==( real_other);
        }
    };

}}}}

#endif

