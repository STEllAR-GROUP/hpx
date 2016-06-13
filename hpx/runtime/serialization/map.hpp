//  Copyright (c) 2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_MAP_HPP
#define HPX_SERIALIZATION_MAP_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/input_archive.hpp>
#include <hpx/runtime/serialization/output_archive.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>

#include <map>
#include <type_traits>

#include <boost/type_traits/add_reference.hpp>
#include <boost/type_traits/remove_const.hpp>

namespace hpx
{
    namespace traits
    {
        template <typename Key, typename Value>
        struct is_bitwise_serializable<std::pair<Key, Value> >
          : std::integral_constant<
                bool,
                is_bitwise_serializable<Key>::value
             && is_bitwise_serializable<Value>::value
            >
        {};
    }

    namespace serialization
    {
        namespace detail
        {
            template <class Key, class Value>
            void load_pair_impl(input_archive& ar, std::pair<Key, Value>& t,
                std::false_type)
            {
                ar >> const_cast<
                    typename boost::add_reference<
                        typename boost::remove_const<Key>::type
                    >::type>(t.first);
                ar >> t.second;
            }

            template <class Key, class Value>
            void load_pair_impl(input_archive& ar, std::pair<Key, Value>& t,
                std::true_type)
            {
                if (ar.disable_array_optimization())
                    load_pair_impl(ar, t, std::false_type());
                else
                    load_binary(ar, &t, sizeof(std::pair<Key, Value>));
            }

            template <class Key, class Value>
            void save_pair_impl(output_archive& ar, const std::pair<Key, Value>& t,
                std::false_type)
            {
                ar << t.first;
                ar << t.second;
            }

            template <class Key, class Value>
            void save_pair_impl(output_archive& ar, const std::pair<Key, Value>& t,
                std::true_type)
            {
                if (ar.disable_array_optimization())
                    save_pair_impl(ar, t, std::false_type());
                else
                    save_binary(ar, &t, sizeof(std::pair<Key, Value>));
            }

        } // namespace detail

        template <class Key, class Value>
        void serialize(input_archive& ar, std::pair<Key, Value>& t, unsigned)
        {
            typedef std::pair<Key, Value> pair_type;
            typedef std::integral_constant<bool,
                hpx::traits::is_bitwise_serializable<pair_type>::value> optimized;

            detail::load_pair_impl(ar, t, optimized());
        }

        template <class Key, class Value>
        void serialize(output_archive& ar, const std::pair<Key, Value>& t, unsigned)
        {
            typedef std::pair<Key, Value> pair_type;
            typedef std::integral_constant<bool,
                hpx::traits::is_bitwise_serializable<pair_type>::value> optimized;

            detail::save_pair_impl(ar, t, optimized());
        }

        template <class Key, class Value, class Comp, class Alloc>
        void serialize(input_archive& ar, std::map<Key, Value, Comp, Alloc>& t, unsigned)
        {
            typedef typename std::map<Key, Value, Comp, Alloc>::size_type size_type;
            typedef typename std::map<Key, Value, Comp, Alloc>::value_type value_type;

            size_type size;
            ar >> size; //-V128

            t.clear();
            for (size_type i = 0; i < size; ++i)
            {
                value_type v;
                ar >> v;
                t.insert(t.end(), std::move(v));
            }
        }

        template <class Key, class Value, class Comp, class Alloc>
        void serialize(output_archive& ar,
            const std::map<Key, Value, Comp, Alloc>& t, unsigned)
        {
            typedef typename std::map<Key, Value, Comp, Alloc>::value_type value_type;

            ar << t.size(); //-V128
            for(const value_type& val : t)
            {
                ar << val;
            }
        }
    }
}

#endif
