//  Copyright (c) 2013 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_STORAGE_TUPLE_HPP_APR_11_2013_1010AM)
#define HPX_UTIL_STORAGE_TUPLE_HPP_APR_11_2013_1010AM

#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/util/any.hpp>
#include <hpx/util/decay.hpp>

#include <type_traits>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace util { namespace storage
{

    ///////////////////////////////////////////////////////////////////////////
    /// This class is an implementation of Tuple.
    class tuple
    {
        public:

            typedef std::vector<hpx::util::any> tuple_holder;
            typedef hpx::util::any elem_type;
            typedef hpx::util::hash_any hash_elem_functor;

            typedef tuple_holder::iterator iterator;
            typedef tuple_holder::const_iterator const_iterator;

            tuple() {}

            ~tuple() {}

            friend class hpx::serialization::access;

            template <typename Archive>
            void serialize(Archive& ar, unsigned const)
            {
                ar & tuple_;
            }

            bool empty() const
            {
                if(tuple_.empty())
                {
                    return true;
                }
                else
                {
                    for(const_iterator it = tuple_.begin();
                            it != tuple_.end(); ++it)
                    {
                        if(!it->empty())
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
                    typename std::enable_if<!std::is_same<
                        elem_type,
                        typename util::decay<T>::type
                    >::value>::type* = nullptr)
            {
                tuple_.push_back(elem_type(field)); // insert an any object
                return *this;
            }

            tuple& push_back_empty()
            {
                tuple_.push_back(elem_type()); // insert an empty any object
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
}}} // hpx::util::storage


#endif

