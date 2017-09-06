//  copyright (c) 2005
//  troy d. straszheim <troy@resophonic.com>
//  http://www.resophonic.com
//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_VARIANT_HPP
#define HPX_SERIALIZATION_VARIANT_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#include <hpx/throw_exception.hpp>

#include <boost/variant.hpp>

#include <utility>

namespace hpx { namespace serialization
{
    struct variant_save_visitor : boost::static_visitor<>
    {
        variant_save_visitor(output_archive& ar)
          : m_ar(ar)
        {}

        template <typename T>
        void operator()(T const& value) const
        {
            m_ar << value;
        }

    private:
        output_archive & m_ar;
    };

    template <typename ... Ts>
    struct variant_impl;

    template <typename T, typename ... Ts>
    struct variant_impl<T, Ts...>
    {
        template <typename V>
        static void load(input_archive& ar, int which, V& v)
        {
            if (which == 0)
            {
            // note: A non-intrusive implementation (such as this one)
            // necessary has to copy the value.  This wouldn't be necessary
            // with an implementation that de-serialized to the address of the
            // aligned storage included in the variant.
                T value;
                ar >> value;
                v = std::move(value);
                return;
            }
            variant_impl<Ts...>::load(ar, which - 1, v);
        }
    };

    template <>
    struct variant_impl<>
    {
        template <typename V>
        static void load(input_archive& /*ar*/, int /*which*/, V& /*v*/)
        {
        }
    };

    template <typename ... T>
    void save(output_archive& ar, boost::variant<T...> const& v, unsigned)
    {
        int which = v.which();
        ar << which;
        variant_save_visitor visitor(ar);
        v.apply_visitor(visitor);
    }

    template <typename ... T>
    void load(input_archive& ar, boost::variant<T...>& v, unsigned)
    {
        int which;
        ar >> which;
        if (which >= static_cast<int>(sizeof...(T)))
        {
            // this might happen if a type was removed from the list of variant
            // types
            HPX_THROW_EXCEPTION(serialization_error
              , "load<Archive, Variant, version>"
              , "type was removed from the list of variant types");
        }
        variant_impl<T...>::load(ar, which, v);
    }

    HPX_SERIALIZATION_SPLIT_FREE_TEMPLATE(
        (template<typename ... T>), (boost::variant<T...>));
}}

#endif //HPX_SERIALIZATION_VARIANT_HPP
