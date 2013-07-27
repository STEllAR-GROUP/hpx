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
#include <boost/mpl/bool.hpp>

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

        ///////////////////////////////////////////////////////////////////////
        template <typename Archive>
        void save(Archive& ar, const unsigned int version) const
        {
            ar << size_;

            typedef typename
                boost::serialization::use_array_optimization<Archive>::template apply<
                    typename boost::remove_const<T>::type
                >::type use_optimized;

            save_optimized(ar, version, use_optimized());
        }

        template <typename Archive>
        void save_optimized(Archive& ar, const unsigned int version, boost::mpl::false_) const
        {
            std::size_t c = size_;
            T* t = data_.get();
            while(c-- > 0)
                ar << *t++;
        }

        template <typename Archive>
        void save_optimized(Archive& ar, const unsigned int version, boost::mpl::true_) const
        {
            boost::serialization::array<T> arr(data_.get(), size_);
            ar.save_array(arr, version);
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Archive>
        void load(Archive& ar, const unsigned int version)
        {
            ar >> size_;
            data_.reset(new T[size_]);

            typedef typename
                boost::serialization::use_array_optimization<Archive>::template apply<
                    typename boost::remove_const<T>::type
                >::type use_optimized;

            load_optimized(ar, version, use_optimized());
        }

        template <typename Archive>
        void load_optimized(Archive& ar, const unsigned int version, boost::mpl::false_)
        {
            std::size_t c = size_;
            T* t = data_.get();
            while(c-- > 0)
                ar >> *t++;
        }

        template <typename Archive>
        void load_optimized(Archive& ar, const unsigned int version, boost::mpl::true_)
        {
            boost::serialization::array<T> arr(data_.get(), size_);
            ar.load_array(arr, version);
        }

        BOOST_SERIALIZATION_SPLIT_MEMBER()

        // this is needed for util::any
        friend bool
        operator==(serialize_buffer const& rhs, serialize_buffer const& lhs)
        {
            return rhs.data_.get() == lhs.data_.get() && rhs.size_ == lhs.size_;
        }

    private:
        boost::shared_ptr<T> data_;
        std::size_t size_;
    };
}}

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for streaming with util::any, we don't want
    // util::serialize_buffer to be streamable
    template <typename T>
    struct supports_streaming_with_any<util::serialize_buffer<T> >
      : boost::mpl::false_
    {};
}}

#endif
