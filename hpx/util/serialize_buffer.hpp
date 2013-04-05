//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_SERIALIZE_BUFFER_APR_05_2013_0312PM)
#define HPX_UTIL_SERIALIZE_BUFFER_APR_05_2013_0312PM

#include <hpx/hpx_fwd.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/split_member.hpp>

#include <algorithm>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class serialize_buffer
    {
    private:
        static void no_deleter(T*) {}

    public:
        enum init_mode
        {
            copy = 0,       // constructor copies data
            reference = 1   // constructor does not copy data and does not 
                            // manage the lifetime of it
        };

        serialize_buffer()
          : size_(0)
        {}

        serialize_buffer (T* data, std::size_t size, init_mode mode = copy)
          : data_(), size_(size)
        {
            if (mode == copy) {
                data_.reset(new T[size]);
                std::copy(data, data + size, data_.get());
            }
            else {
                data_ = boost::shared_ptr<T>(data, &serialize_buffer::no_deleter);
            }
        }

        void const* data() const { return data_; }
        std::size_t size() const { return size_; }

    private:
        // serialization support
        friend class boost::serialization::access;

        template <typename Archive>
        void save(Archive& ar, const unsigned int version) const
        {
            ar << size_;
            boost::serialization::array<T> arr(data_.get(), size_);
            ar << arr;
        }

        template <typename Archive>
        void load(Archive& ar, const unsigned int version)
        {
            ar >> size_;
            data_.reset(new T[size_]);

            boost::serialization::array<T> arr(data_.get(), size_);
            ar >> arr;
        }

        BOOST_SERIALIZATION_SPLIT_MEMBER()

    private:
        boost::shared_ptr<T> data_;
        std::size_t size_;
    };
}}

#endif
