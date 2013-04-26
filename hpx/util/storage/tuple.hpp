//  Copyright (c) 2013 Shuangyang Yang
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_STORAGE_TUPLE_HPP_APR_11_2013_1010AM)
#define HPX_UTIL_STORAGE_TUPLE_HPP_APR_11_2013_1010AM

#include <vector>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/access.hpp>

#include <hpx/util/any.hpp>

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
            typedef std::string key_type;

            typedef tuple_holder::iterator iterator;
            typedef tuple_holder::const_iterator const_iterator;

            tuple() {}

            ~tuple() {}

            friend class boost::serialization::access;

            template <typename Archive>
            void serialize(Archive& ar, unsigned const)
            {
                ar & tuple_;
            }

            bool empty() const
            {
                return tuple_.empty();
            }

            template <typename T>
            tuple& push_back(const T& field)
            {
                tuple_.push_back(elem_type(field)); // insert an any object
                return *this;
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

