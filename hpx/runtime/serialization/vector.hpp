//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_VECTOR_HPP
#define HPX_SERIALIZATION_VECTOR_HPP

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/traits/is_bitwise_serializable.hpp>

#include <vector>

namespace hpx { namespace serialization
{
    // load vector<T>
    template <typename T, typename Allocator>
    void load_impl(input_archive & ar, std::vector<T, Allocator> & vs, boost::mpl::false_)
    {
        // normal load ...
        typedef typename std::vector<T>::size_type size_type;
        size_type size;
        ar >> size; //-V128
        if(size == 0) return;

        vs.reserve(size);
        typedef typename std::vector<T>::value_type value_type;
        for(size_type i = 0; i != size; ++i)
        {
            value_type v;
            ar >> v;
            vs.push_back(std::move(v));
        }
    }

    template <typename T, typename Allocator>
    void load_impl(input_archive & ar, std::vector<T, Allocator> & v, boost::mpl::true_)
    {
        if(!has_array_optimization(ar))
        {
            load_impl(ar, v, boost::mpl::false_());
        }
        else
        {
            // bitwise load ...
            typedef typename std::vector<T>::value_type value_type;
            typedef typename std::vector<T>::size_type size_type;
            size_type size;
            ar >> size; //-V128
            if(size == 0) return;

            v.resize(size);
            load_binary(ar, &v[0], v.size() * sizeof(value_type));
        }
    }

    template <typename Allocator>
    void serialize(input_archive & ar, std::vector<bool, Allocator> & v, unsigned)
    {
        typedef typename std::vector<bool>::size_type size_type;
        size_type size = 0;
        ar >> size; //-V128
        if(size == 0) return;
        v.clear();

        v.reserve(size);
        // normal load ... no chance of doing bitwise here ...
        for(size_type i = 0; i != size; ++i)
        {
            bool b = false;
            ar >> b;
            v.push_back(b);
        }
    }
    template <typename T, typename Allocator>
    void serialize(input_archive & ar, std::vector<T, Allocator> & v, unsigned)
    {
        v.clear();
        load_impl(
            ar
          , v
          , typename traits::is_bitwise_serializable<
                typename std::vector<T>::value_type
            >::type()
        );
    }

    // save vector<T>
    template <typename T, typename Allocator>
    void save_impl(output_archive & ar, std::vector<T, Allocator> & vs, boost::mpl::false_)
    {
        // normal save ...
        typedef typename std::vector<T>::value_type value_type;
        for(value_type & v : vs)
        {
            ar << v;
        }
    }

    template <typename T, typename Allocator>
    void save_impl(output_archive & ar, std::vector<T, Allocator> & v, boost::mpl::true_)
    {
        if(!has_array_optimization(ar))
        {
            save_impl(ar, v, boost::mpl::false_());
        }
        else
        {
            // bitwise save ...
            typedef typename std::vector<T>::value_type value_type;
            save_binary(ar, &v[0], v.size() * sizeof(value_type));
        }
    }

    template <typename Allocator>
    void serialize(output_archive & ar, std::vector<bool, Allocator> & v, unsigned)
    {
        typedef typename std::vector<bool>::size_type size_type;
        ar << v.size(); //-V128
        if(v.empty()) return;
        // normal save ... no chance of doing bitwise here ...
        for(size_type i = 0; i < v.size(); ++i)
        {
            bool b = v[i];
            ar << b;
        }
    }

    template <typename T, typename Allocator>
    void serialize(output_archive & ar, std::vector<T, Allocator> & v, unsigned)
    {
        ar << v.size(); //-V128
        if(v.empty()) return;
        save_impl(
            ar
          , v
          , typename traits::is_bitwise_serializable<
                typename std::vector<T>::value_type
            >::type()
        );
    }
}}

#endif
