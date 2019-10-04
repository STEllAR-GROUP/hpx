//  Copyright (c) 2013 Shuangyang Yang
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_STORAGE_TUPLE_HPP_APR_11_2013_1010AM)
#define HPX_UTIL_STORAGE_TUPLE_HPP_APR_11_2013_1010AM

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/type_support/decay.hpp>
#include <hpx/util/serializable_any.hpp>

#include <type_traits>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace storage {

    ///////////////////////////////////////////////////////////////////////////
    /// This class is an implementation of Tuple.
    class tuple
    {
    private:
        struct compare_any
        {
            bool operator()(
                hpx::util::any const& lhs, hpx::util::any const& rhs) const
            {
                return lhs.equal_to(rhs);
            }
        };

    public:
        using tuple_holder = std::vector<hpx::util::any>;
        using elem_type = hpx::util::any;
        using hash_elem_functor = hpx::util::hash_any;
        using compare_elem_functor = compare_any;

        using iterator = tuple_holder::iterator;
        using const_iterator = tuple_holder::const_iterator;

        tuple() = default;
        ~tuple() = default;

        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, unsigned const)
        {
            ar & tuple_;
        }

        bool empty() const
        {
            if (tuple_.empty())
            {
                return true;
            }
            else
            {
                for (const auto& it : tuple_)
                {
                    if (it.has_value())
                    {
                        return false;
                    }
                }
                return true;
            }
        }

        size_t size() const
        {
            return tuple_.size();
        }

        tuple& push_back(const elem_type& elem)
        {
            tuple_.push_back(elem);
            return *this;
        }

        template <typename T>
        tuple& push_back(const T& field,
            typename std::enable_if<!std::is_same<elem_type,
                typename util::decay<T>::type>::value>::type* = nullptr)
        {
            tuple_.push_back(elem_type(field));    // insert an any object
            return *this;
        }

        tuple& push_back_empty()
        {
            tuple_.emplace_back();    // insert an empty any object
            return *this;
        }

        template <typename T>
        T get(unsigned int index)
        {
            return hpx::util::any_cast<T>(tuple_.at(index));
        }

        iterator begin()
        {
            return tuple_.begin();
        }

        const_iterator begin() const
        {
            return tuple_.begin();
        }

        iterator end()
        {
            return tuple_.end();
        }

        const_iterator end() const
        {
            return tuple_.end();
        }

    private:
        tuple_holder tuple_;
    };
}}}    // namespace hpx::util::storage

#endif
