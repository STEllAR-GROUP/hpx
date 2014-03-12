//  Copyright (c) 2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_SERIALIZE_BUFFER_APR_05_2013_0312PM)
#define HPX_UTIL_SERIALIZE_BUFFER_APR_05_2013_0312PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/util/bind.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/shared_array.hpp>
#include <boost/mpl/bool.hpp>

#include <hpx/util/serialize_allocator.hpp>

#include <algorithm>

namespace hpx { namespace util
{
    namespace detail
    {
        struct serialize_buffer_no_allocator {};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Allocator = detail::serialize_buffer_no_allocator>
    class serialize_buffer
    {
    private:
        typedef Allocator allocator_type;

        static void no_deleter(T*) {}

        static void deleter(T* p, Allocator alloc, std::size_t size)
        {
            alloc.deallocate(p, size);
        }

    public:
        enum init_mode
        {
            copy = 0,       // constructor copies data
            reference = 1,  // constructor does not copy data and does not
                            // manage the lifetime of it
            take = 2        // constructor does not copy data but does take
                            // ownership and manages the lifetime of it
        };

        explicit serialize_buffer(allocator_type const& alloc = Allocator())
          : size_(0)
          , alloc_(alloc)
        {}

        serialize_buffer (T const* data, std::size_t size,
                allocator_type const& alloc = Allocator())
          : data_()
          , size_(size)
          , alloc_(alloc)
        {
            // create from const data implies 'copy' mode
            using util::placeholders::_1;
            data_.reset(alloc_.allocate(size),
                util::bind(&serialize_buffer::deleter, _1, alloc_, size_));
            if (size != 0)
                std::copy(data, data + size, data_.get());
        }

        serialize_buffer (T* data, std::size_t size, init_mode mode,
                allocator_type const& alloc = Allocator())
          : data_()
          , size_(size)
          , alloc_(alloc)
        {
            if (mode == copy) {
                using util::placeholders::_1;
                data_.reset(alloc_.allocate(size),
                    util::bind(&serialize_buffer::deleter, _1, alloc_, size_));
                if (size != 0)
                    std::copy(data, data + size, data_.get());
            }
            else if (mode == reference) {
                data_ = boost::shared_array<T>(data,
                    &serialize_buffer::no_deleter);
            }
            else {
                // take ownership
                using util::placeholders::_1;
                data_ = boost::shared_array<T>(data,
                    util::bind(&serialize_buffer::deleter, _1, alloc_, size_));
            }
        }

        T* data() { return data_.get(); }
        T const* data() const { return data_.get(); }

        std::size_t size() const { return size_; }

    private:
        // serialization support
        friend class boost::serialization::access;

        ///////////////////////////////////////////////////////////////////////
        template <typename Archive>
        void save(Archive& ar, const unsigned int version) const
        {
            ar << size_ << alloc_;

            typedef typename
                boost::serialization::use_array_optimization<Archive>::template apply<
                    typename boost::remove_const<T>::type
                >::type use_optimized;

            save_optimized(ar, version, use_optimized());
        }

        template <typename Archive>
        void save_optimized(Archive& ar, const unsigned int, boost::mpl::false_) const
        {
            std::size_t c = size_;
            T* t = data_.get();
            while(c-- > 0)
                ar << *t++;
        }

        template <typename Archive>
        void save_optimized(Archive& ar, const unsigned int version, boost::mpl::true_) const
        {
            if (size_ != 0)
            {
                boost::serialization::array<T> arr(data_.get(), size_);
                ar.save_array(arr, version);
            }
        }

        ///////////////////////////////////////////////////////////////////////
        template <typename Archive>
        void load(Archive& ar, const unsigned int version)
        {
            using util::placeholders::_1;
            ar >> size_ >> alloc_;

            data_.reset(alloc_.allocate(size_),
                util::bind(&serialize_buffer::deleter, _1, alloc_, size_));

            typedef typename
                boost::serialization::use_array_optimization<Archive>::template apply<
                    typename boost::remove_const<T>::type
                >::type use_optimized;

            load_optimized(ar, version, use_optimized());
        }

        template <typename Archive>
        void load_optimized(Archive& ar, const unsigned int, boost::mpl::false_)
        {
            std::size_t c = size_;
            T* t = data_.get();
            while(c-- > 0)
                ar >> *t++;
        }

        template <typename Archive>
        void load_optimized(Archive& ar, const unsigned int version, boost::mpl::true_)
        {
            if (size_ != 0)
            {
                boost::serialization::array<T> arr(data_.get(), size_);
                ar.load_array(arr, version);
            }
        }

        BOOST_SERIALIZATION_SPLIT_MEMBER()

        // this is needed for util::any
        friend bool
        operator==(serialize_buffer const& rhs, serialize_buffer const& lhs)
        {
            return rhs.data_.get() == lhs.data_.get() && rhs.size_ == lhs.size_;
        }

    private:
        boost::shared_array<T> data_;
        std::size_t size_;
        Allocator alloc_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class serialize_buffer<T, detail::serialize_buffer_no_allocator>
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

        serialize_buffer (T const* data, std::size_t size)
          : data_(), size_(size)
        {
            // create from const data implies 'copy' mode
            data_.reset(new T[size]);
            if (size != 0)
                std::copy(data, data + size, data_.get());
        }

        serialize_buffer (T* data, std::size_t size, init_mode mode)
          : data_(), size_(size)
        {
            if (mode == copy) {
                data_.reset(new T[size]);
                if (size != 0)
                    std::copy(data, data + size, data_.get());
            }
            else if (mode == reference) {
                data_ = boost::shared_array<T>(data,
                    &serialize_buffer::no_deleter);
            }
            else {
                // take ownership
                data_ = boost::shared_array<T>(data);
            }
        }

        T* data() { return data_.get(); }
        T const* data() const { return data_.get(); }

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
        void save_optimized(Archive& ar, const unsigned int, boost::mpl::false_) const
        {
            std::size_t c = size_;
            T* t = data_.get();
            while(c-- > 0)
                ar << *t++;
        }

        template <typename Archive>
        void save_optimized(Archive& ar, const unsigned int version, boost::mpl::true_) const
        {
            if (size_ != 0)
            {
                boost::serialization::array<T> arr(data_.get(), size_);
                ar.save_array(arr, version);
            }
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
        void load_optimized(Archive& ar, const unsigned int, boost::mpl::false_)
        {
            std::size_t c = size_;
            T* t = data_.get();
            while(c-- > 0)
                ar >> *t++;
        }

        template <typename Archive>
        void load_optimized(Archive& ar, const unsigned int version, boost::mpl::true_)
        {
            if (size_ != 0)
            {
                boost::serialization::array<T> arr(data_.get(), size_);
                ar.load_array(arr, version);
            }
        }

        BOOST_SERIALIZATION_SPLIT_MEMBER()

        // this is needed for util::any
        friend bool
        operator==(serialize_buffer const& rhs, serialize_buffer const& lhs)
        {
            return rhs.data_.get() == lhs.data_.get() && rhs.size_ == lhs.size_;
        }

    private:
        boost::shared_array<T> data_;
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

    template <typename T>
    struct type_size<util::serialize_buffer<T> >
    {
        static std::size_t call(util::serialize_buffer<T> const& buffer_)
        {
            return buffer_.size() * sizeof(T);
        }
    };
}}

#endif
