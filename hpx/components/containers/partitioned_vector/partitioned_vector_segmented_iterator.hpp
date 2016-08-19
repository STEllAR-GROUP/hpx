//  Copyright (c) 2014 Anuj R. Sharma
//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http:// ww.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARTITIONED_VECTOR_SEGMENTED_ITERATOR_HPP
#define HPX_PARTITIONED_VECTOR_SEGMENTED_ITERATOR_HPP

/// \file hpx/components/partitioned_vector/partitioned_vector_segmented_iterator.hpp
/// \brief This file contains the implementation of iterators for hpx::partitioned_vector.

 // The idea for these iterators is taken from
 // http://lafstern.org/matt/segmented.pdf.

#include <hpx/config.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/traits/segmented_iterator_traits.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/iterator_adaptor.hpp>
#include <hpx/util/iterator_facade.hpp>

#include <hpx/components/containers/partitioned_vector/partitioned_vector_fwd.hpp>
#include <hpx/components/containers/partitioned_vector/partitioned_vector_component.hpp>

#include <boost/integer.hpp>

#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace hpx
{
    ///////////////////////////////////////////////////////////////////////////
    // This class wraps plain a vector<>::iterator or vector<>::const_iterator
    template <typename T, typename Data, typename BaseIter>
    class local_raw_vector_iterator
      : public hpx::util::iterator_adaptor<
            local_raw_vector_iterator<T, Data, BaseIter>, BaseIter
        >
    {
    private:
        typedef hpx::util::iterator_adaptor<
                local_raw_vector_iterator<T, Data, BaseIter>, BaseIter
            > base_type;
        typedef BaseIter base_iterator;

    public:
        typedef local_vector_iterator<T, Data> local_iterator;
        typedef const_local_vector_iterator<T, Data> local_const_iterator;

        local_raw_vector_iterator() {}

        local_raw_vector_iterator(base_iterator const& it,
                std::shared_ptr<server::partitioned_vector<T, Data> > const& data)
          : base_type(it), data_(data)
        {}

        local_iterator remote()
        {
            HPX_ASSERT(data_);
            std::size_t local_index = std::distance(data_->begin(), this->base());
            return local_iterator(
                partitioned_vector_partition<T>(data_->get_id()),
                local_index, data_);
        }
        local_const_iterator remote() const
        {
            HPX_ASSERT(data_);
            std::size_t local_index = std::distance(data_->begin(), this->base());
            return local_const_iterator(
                partitioned_vector_partition<T>(data_->get_id()),
                local_index, data_);
        }

    private:
        std::shared_ptr<server::partitioned_vector<T, Data> > data_;
    };

    template <typename T, typename Data, typename BaseIter>
    class const_local_raw_vector_iterator
      : public hpx::util::iterator_adaptor<
            const_local_raw_vector_iterator<T, Data, BaseIter>, BaseIter
        >
    {
    private:
        typedef hpx::util::iterator_adaptor<
                const_local_raw_vector_iterator<T, Data, BaseIter>, BaseIter
            > base_type;
        typedef BaseIter base_iterator;

    public:
        typedef const_local_vector_iterator<T, Data> local_iterator;
        typedef const_local_vector_iterator<T, Data> local_const_iterator;

        const_local_raw_vector_iterator() {}

        const_local_raw_vector_iterator(base_iterator const& it,
                std::shared_ptr<server::partitioned_vector<T, Data> > const& data)
          : base_type(it), data_(data)
        {}

        local_const_iterator remote()
        {
            HPX_ASSERT(data_);
            std::size_t local_index = std::distance(data_->cbegin(), this->base());
            return local_const_iterator(
                partitioned_vector_partition<T>(data_->get_id()),
                local_index, data_);
        }
        local_const_iterator remote() const
        {
            HPX_ASSERT(data_);
            std::size_t local_index = std::distance(data_->cbegin(), this->base());
            return local_const_iterator(
                partitioned_vector_partition<T>(data_->get_id()),
                local_index, data_);
        }

    private:
        std::shared_ptr<server::partitioned_vector<T, Data> > data_;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Data>
        struct local_vector_value_proxy
        {
            local_vector_value_proxy(local_vector_iterator<T, Data> const& it)
              : it_(it)
            {}

            operator T() const
            {
                if (!it_.get_data())
                {
                    return it_.get_partition().get_value(launch::sync,
                        it_.get_local_index());
                }
                return *(it_.get_data()->begin() + it_.get_local_index());
            }

            template <typename T_>
            local_vector_value_proxy& operator=(T_ && value)
            {
                if (!it_.get_data())
                {
                    it_.get_partition().set_value(launch::sync,
                        it_.get_local_index(), std::forward<T_>(value));
                }
                else
                {
                    *(it_.get_data()->begin() + it_.get_local_index()) =
                        std::forward<T_>(value);
                }
                return *this;
            }

            local_vector_iterator<T, Data> const& it_;
        };

        template <typename T, typename Data>
        struct const_local_vector_value_proxy
        {
            const_local_vector_value_proxy(
                    const_local_vector_iterator<T, Data> const& it)
              : it_(it)
            {}

            operator T() const
            {
                if (!it_.get_data())
                {
                    return it_.get_partition().get_value(launch::sync,
                        it_.get_local_index());
                }
                return *it_.local();
            }

            const_local_vector_iterator<T, Data> const& it_;
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename T, typename Data>
        struct vector_value_proxy
        {
            vector_value_proxy(hpx::partitioned_vector<T, Data>& v,
                    std::size_t index)
              : v_(v), index_(index)
            {}

            operator T() const
            {
                return v_.get_value(launch::sync, index_);
            }

            template <typename T_>
            vector_value_proxy& operator=(T_ && value)
            {
                v_.set_value(launch::sync, index_, std::forward<T_>(value));
                return *this;
            }

            partitioned_vector<T, Data>& v_;
            std::size_t index_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    /// This class implements the local iterator functionality for the
    /// partitioned backend of a hpx::vector.
    template <typename T, typename Data>
    class local_vector_iterator
      : public hpx::util::iterator_facade<
            local_vector_iterator<T, Data>, T,
            std::random_access_iterator_tag,
            detail::local_vector_value_proxy<T, Data>
        >
    {
    private:
        typedef hpx::util::iterator_facade<
                local_vector_iterator<T, Data>, T,
                std::random_access_iterator_tag,
                detail::local_vector_value_proxy<T, Data>
            > base_type;

    public:
        typedef std::size_t size_type;

        // constructors
        local_vector_iterator()
          : partition_(), local_index_(size_type(-1))
        {}

        local_vector_iterator(partitioned_vector_partition<T, Data> partition,
                size_type local_index,
                std::shared_ptr<server::partitioned_vector<T, Data> > const& data)
          : partition_(partition),
            local_index_(local_index),
            data_(data)
        {}

        typedef local_raw_vector_iterator<
                T, Data, typename Data::iterator
            > local_raw_iterator;
        typedef const_local_raw_vector_iterator<
                T, Data, typename Data::const_iterator
            > local_raw_const_iterator;

        ///////////////////////////////////////////////////////////////////////
        local_raw_iterator local()
        {
            if (partition_ && !data_)
                data_ = partition_.get_ptr();
            return local_raw_iterator(data_->begin() + local_index_, data_);
        }
        local_raw_const_iterator local() const
        {
            if (partition_ && !data_)
                data_ = partition_.get_ptr();
            return local_raw_iterator(data_->cbegin() + local_index_, data_);
        }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void load(Archive& ar, unsigned version)
        {
            ar & partition_ & local_index_;
        }
        template <typename Archive>
        void save(Archive& ar, unsigned version) const
        {
            ar & partition_ & local_index_;
        }

        HPX_SERIALIZATION_SPLIT_MEMBER()

    protected:
        friend class hpx::util::iterator_core_access;

        bool equal(local_vector_iterator const& other) const
        {
            return partition_ == other.partition_ &&
                local_index_ == other.local_index_;
        }

        typename base_type::reference dereference() const
        {
            return detail::local_vector_value_proxy<T, Data>(*this);
        }

        void increment()
        {
            ++local_index_;
        }

        void decrement()
        {
            --local_index_;
        }

        void advance(std::ptrdiff_t n)
        {
            local_index_ += n;
        }

        std::ptrdiff_t distance_to(local_vector_iterator const& other) const
        {
            HPX_ASSERT(partition_ == other.partition_);
            return other.local_index_ - local_index_;
        }

    public:
        partitioned_vector_partition<T>& get_partition() { return partition_; }
        partitioned_vector_partition<T> get_partition() const { return partition_; }

        size_type get_local_index() const { return local_index_; }

        std::shared_ptr<server::partitioned_vector<T, Data> >& get_data()
        {
            if (partition_ && !data_)
                data_ = partition_.get_ptr();
            return data_;
        }
        std::shared_ptr<server::partitioned_vector<T, Data> > const&
            get_data() const
        {
            if (partition_ && !data_)
                data_ = partition_.get_ptr();
            return data_;
        }

    protected:
        // refer to a partition of the vector
        partitioned_vector_partition<T, Data> partition_;

        // local position in the referenced partition
        size_type local_index_;

        // caching address of component
        mutable std::shared_ptr<server::partitioned_vector<T, Data> > data_;
    };

    template <typename T, typename Data>
    class const_local_vector_iterator
      : public hpx::util::iterator_facade<
            const_local_vector_iterator<T, Data>, T const,
            std::random_access_iterator_tag,
            detail::const_local_vector_value_proxy<T, Data>
        >
    {
    private:
        typedef hpx::util::iterator_facade<
                const_local_vector_iterator<T, Data>, T const,
                std::random_access_iterator_tag,
                detail::const_local_vector_value_proxy<T, Data>
            > base_type;

    public:
        typedef std::size_t size_type;

        // constructors
        const_local_vector_iterator()
          : partition_(), local_index_(size_type(-1))
        {}

        const_local_vector_iterator(
                partitioned_vector_partition<T, Data> partition,
                size_type local_index,
                std::shared_ptr<server::partitioned_vector<T, Data> > const& data)
          : partition_(partition),
            local_index_(local_index),
            data_(data)
        {}

        const_local_vector_iterator(local_vector_iterator<T, Data> it)
          : partition_(it.get_partition()),
            local_index_(it.get_local_index()),
            data_(it.get_data())
        {}

        typedef const_local_raw_vector_iterator<
                T, Data, typename Data::const_iterator
            > local_raw_iterator;
        typedef local_raw_iterator local_raw_const_iterator;

        ///////////////////////////////////////////////////////////////////////
        local_raw_iterator local()
        {
            if (partition_ && !data_)
                data_ = partition_.get_ptr();
            return local_raw_iterator(data_->cbegin() + local_index_, data_);
        }
        local_raw_const_iterator local() const
        {
            if (partition_ && !data_)
                data_ = partition_.get_ptr();
            return local_raw_const_iterator(data_->cbegin() + local_index_, data_);
        }

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void load(Archive& ar, unsigned version)
        {
            ar & partition_ & local_index_;
        }
        template <typename Archive>
        void save(Archive& ar, unsigned version) const
        {
            ar & partition_ & local_index_;
        }

        HPX_SERIALIZATION_SPLIT_MEMBER()

    protected:
        friend class hpx::util::iterator_core_access;

        bool equal(const_local_vector_iterator const& other) const
        {
            return partition_ == other.partition_ &&
                local_index_ == other.local_index_;
        }

        typename base_type::reference dereference() const
        {
            return detail::const_local_vector_value_proxy<T, Data>(*this);
        }

        void increment()
        {
            ++local_index_;
        }

        void decrement()
        {
            --local_index_;
        }

        void advance(std::ptrdiff_t n)
        {
            local_index_ += n;
        }

        std::ptrdiff_t distance_to(const_local_vector_iterator const& other) const
        {
            HPX_ASSERT(partition_ == other.partition_);
            return other.local_index_ - local_index_;
        }

    public:
        partitioned_vector_partition<T> const& get_partition() const
        {
            return partition_;
        }
        size_type get_local_index() const { return local_index_; }

        std::shared_ptr<server::partitioned_vector<T, Data> >& get_data()
        {
            if (partition_ && !data_)
                data_ = partition_.get_ptr();
            return data_;
        }
        std::shared_ptr<server::partitioned_vector<T, Data> > const&
            get_data() const
        {
            if (partition_ && !data_)
                data_ = partition_.get_ptr();
            return data_;
        }

    protected:
        // refer to a partition of the vector
        partitioned_vector_partition<T, Data> partition_;

        // local position in the referenced partition
        size_type local_index_;

        // caching address of component
        mutable std::shared_ptr<server::partitioned_vector<T, Data> > data_;
    };

    ///////////////////////////////////////////////////////////////////////////
    /// This class implement the segmented iterator for the hpx::vector
    template <typename T, typename Data, typename BaseIter>
    class segment_vector_iterator
      : public hpx::util::iterator_adaptor<
            segment_vector_iterator<T, Data, BaseIter>, BaseIter
        >
    {
    private:
        typedef hpx::util::iterator_adaptor<
                segment_vector_iterator<T, Data, BaseIter>, BaseIter
            > base_type;

    public:
        segment_vector_iterator()
          : data_(0)
        {}

        segment_vector_iterator(BaseIter const& it,
                partitioned_vector<T, Data>* data = nullptr)
          : base_type(it), data_(data)
        {}

        partitioned_vector<T, Data>* get_data() { return data_; }
        partitioned_vector<T, Data> const* get_data() const { return data_; }

        bool is_at_end() const
        {
            return data_ == nullptr ||
                this->base_type::base_reference() == data_->partitions_.end();
        }

    private:
        partitioned_vector<T, Data>* data_;
    };

    template <typename T, typename Data, typename BaseIter>
    class const_segment_vector_iterator
      : public hpx::util::iterator_adaptor<
            const_segment_vector_iterator<T, Data, BaseIter>, BaseIter
        >
    {
    private:
        typedef hpx::util::iterator_adaptor<
                const_segment_vector_iterator<T, Data, BaseIter>, BaseIter
            > base_type;

    public:
        const_segment_vector_iterator()
          : data_(0)
        {}

        const_segment_vector_iterator(BaseIter const& it,
                partitioned_vector<T, Data> const* data = nullptr)
          : base_type(it), data_(data)
        {}

        partitioned_vector<T, Data> const* get_data() const { return data_; }

        bool is_at_end() const
        {
            return data_ == nullptr ||
                this->base_type::base_reference() == data_->partitions_.end();
        }

    private:
        partitioned_vector<T, Data> const* data_;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        template <typename BaseIterator>
        struct is_requested_locality
        {
            typedef typename std::iterator_traits<BaseIterator>::reference
                reference;

            is_requested_locality(boost::uint32_t locality_id =
                    naming::invalid_locality_id)
              : locality_id_(locality_id)
            {}

            bool operator()(reference val) const
            {
                return locality_id_ == naming::invalid_locality_id ||
                       locality_id_ == val.locality_id_;
            }

            boost::uint32_t locality_id_;
        };
    }

    /// This class implement the local segmented iterator for the hpx::vector
    template <typename T, typename Data, typename BaseIter>
    class local_segment_vector_iterator
      : public hpx::util::iterator_adaptor<
            local_segment_vector_iterator<T, Data, BaseIter>, BaseIter,
            Data, std::forward_iterator_tag
        >
    {
    private:
        typedef hpx::util::iterator_adaptor<
                local_segment_vector_iterator<T, Data, BaseIter>, BaseIter,
                Data, std::forward_iterator_tag
            > base_type;
        typedef detail::is_requested_locality<BaseIter> predicate;

    public:
        local_segment_vector_iterator()
        {}

        local_segment_vector_iterator(BaseIter const& end)
          : base_type(end), predicate_(), end_(end)
        {}

        local_segment_vector_iterator(
                BaseIter const& it, BaseIter const& end,
                boost::uint32_t locality_id)
          : base_type(it), predicate_(locality_id), end_(end)
        {
            satisfy_predicate();
        }

        bool is_at_end() const
        {
            return !data_ || this->base() == end_;
        }

    private:
        friend class hpx::util::iterator_core_access;

        typename base_type::reference dereference() const
        {
            HPX_ASSERT(!is_at_end());
            return data_->get_data();
        }

        void increment()
        {
            ++(this->base_reference());
            satisfy_predicate();
        }

        void satisfy_predicate()
        {
            while (this->base() != end_ && !predicate_(*this->base()))
                ++(this->base_reference());

            if (this->base() != end_)
                data_ = this->base()->local_data_;
            else
                data_.reset();
        }

    private:
        std::shared_ptr<server::partitioned_vector<T, Data> > data_;
        predicate predicate_;
        BaseIter end_;
    };

    ///////////////////////////////////////////////////////////////////////////
    /// This class implements the (global) iterator functionality for hpx::vector.
    template <typename T, typename Data>
    class vector_iterator
      : public hpx::util::iterator_facade<
            vector_iterator<T, Data>, T, std::random_access_iterator_tag,
            detail::vector_value_proxy<T, Data>
        >
    {
    private:
        typedef hpx::util::iterator_facade<
                vector_iterator<T, Data>, T, std::random_access_iterator_tag,
                detail::vector_value_proxy<T, Data>
            > base_type;

    public:
        typedef std::size_t size_type;
        typedef typename partitioned_vector<T, Data>::segment_iterator
            segment_iterator;
        typedef typename partitioned_vector<T, Data>::local_segment_iterator
            local_segment_iterator;
        typedef typename partitioned_vector<T, Data>::local_iterator
            local_iterator;

        // constructors
        vector_iterator()
          : data_(0), global_index_(size_type(-1))
        {}

        vector_iterator(partitioned_vector<T, Data>* data,
                size_type global_index)
          : data_(data), global_index_(global_index)
        {}

        partitioned_vector<T, Data>* get_data() { return data_; }
        partitioned_vector<T, Data> const* get_data() const { return data_; }

        size_type get_global_index() const { return global_index_; }

    protected:
        friend class hpx::util::iterator_core_access;

        bool equal(vector_iterator const& other) const
        {
            return data_ == other.data_ && global_index_ == other.global_index_;
        }

        typename base_type::reference dereference() const
        {
            HPX_ASSERT(data_);
            return detail::vector_value_proxy<T, Data>(*data_, global_index_);
        }

        void increment()
        {
            HPX_ASSERT(data_);
            ++global_index_;
        }

        void decrement()
        {
            HPX_ASSERT(data_);
            --global_index_;
        }

        void advance(std::ptrdiff_t n)
        {
            HPX_ASSERT(data_);
            global_index_ += n;
        }

        std::ptrdiff_t distance_to(vector_iterator const& other) const
        {
            HPX_ASSERT(data_ && other.data_);
            HPX_ASSERT(data_ == other.data_);
            return other.global_index_ - global_index_;
        }

    protected:
        // refer to the vector
        partitioned_vector<T, Data>* data_;

        // global position in the referenced vector
        size_type global_index_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Data>
    class const_vector_iterator
      : public hpx::util::iterator_facade<
            const_vector_iterator<T, Data>, T const,
            std::random_access_iterator_tag, T const
        >
    {
    private:
        typedef hpx::util::iterator_facade<
                const_vector_iterator<T, Data>, T const,
                std::random_access_iterator_tag, T const
            > base_type;

    public:
        typedef std::size_t size_type;
        typedef typename partitioned_vector<T, Data>::const_segment_iterator
            segment_iterator;
        typedef typename partitioned_vector<T, Data>::const_local_segment_iterator
            local_segment_iterator;
        typedef typename partitioned_vector<T, Data>::const_local_iterator
            local_iterator;

        // constructors
        const_vector_iterator()
          : data_(0), global_index_(size_type(-1))
        {}

        const_vector_iterator(partitioned_vector<T, Data> const* data,
                size_type global_index)
          : data_(data), global_index_(global_index)
        {}

        partitioned_vector<T, Data> const* get_data() const { return data_; }
        size_type get_global_index() const { return global_index_; }

    protected:
        friend class hpx::util::iterator_core_access;

        bool equal(const_vector_iterator const& other) const
        {
            return data_ == other.data_ && global_index_ == other.global_index_;
        }

        typename base_type::reference dereference() const
        {
            HPX_ASSERT(data_);
            return data_->get_value(launch::sync, global_index_);
        }

        void increment()
        {
            HPX_ASSERT(data_);
            ++global_index_;
        }

        void decrement()
        {
            HPX_ASSERT(data_);
            --global_index_;
        }

        void advance(std::ptrdiff_t n)
        {
            HPX_ASSERT(data_);
            global_index_ += n;
        }

        std::ptrdiff_t distance_to(const_vector_iterator const& other) const
        {
            HPX_ASSERT(data_ && other.data_);
            HPX_ASSERT(data_ == other.data_);
            return other.global_index_ - global_index_;
        }

    protected:
        // refer to the vector
        partitioned_vector<T, Data> const* data_;

        // global position in the referenced vector
        size_type global_index_;
    };
}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    template <typename T, typename Data>
    struct segmented_iterator_traits<vector_iterator<T, Data> >
    {
        typedef std::true_type is_segmented_iterator;

        typedef vector_iterator<T, Data> iterator;
        typedef typename iterator::segment_iterator segment_iterator;
        typedef typename iterator::local_segment_iterator local_segment_iterator;
        typedef typename iterator::local_iterator local_iterator;

        typedef typename local_iterator::local_raw_iterator local_raw_iterator;

        //  Conceptually this function is supposed to denote which segment
        //  the iterator is currently pointing to (i.e. just global iterator).
        static segment_iterator segment(iterator iter)
        {
            return iter.get_data()->get_segment_iterator(
                iter.get_global_index());
        }

        //  This function should specify which is the current segment and
        //  the exact position to which local iterator is pointing.
        static local_iterator local(iterator iter)
        {
            HPX_ASSERT(iter.get_data());    // avoid dereferencing end iterator
            return iter.get_data()->get_local_iterator(
                iter.get_global_index());
        }

        //  Build a full iterator from the segment and local iterators
        static iterator compose(segment_iterator seg_iter,
            local_iterator local_iter)
        {
            partitioned_vector<T, Data>* data = seg_iter.get_data();
            std::size_t index = local_iter.get_local_index();
            return iterator(data, data->get_global_index(seg_iter, index));
        }

        //  This function should specify the local iterator which is at the
        //  beginning of the partition.
        static local_iterator begin(segment_iterator seg_iter)
        {
            std::size_t offset = 0;
            if (seg_iter.is_at_end())
            {
                // return iterator to the end of last segment
                --seg_iter;
                offset = seg_iter.base()->size_;
            }

            return local_iterator(seg_iter.base()->partition_, offset,
                seg_iter.base()->local_data_);
        }

        //  This function should specify the local iterator which is at the
        //  end of the partition.
        static local_iterator end(segment_iterator seg_iter)
        {
            if (seg_iter.is_at_end())
                --seg_iter;     // return iterator to the end of last segment

            auto& base = seg_iter.base();
            return local_iterator(base->partition_, base->size_,
                base->local_data_);
        }

        //  This function should specify the local iterator which is at the
        //  beginning of the partition data.
        static local_raw_iterator begin(local_segment_iterator const& seg_iter)
        {
            return local_raw_iterator(seg_iter->begin(),
                seg_iter.base()->local_data_);
        }

        //  This function should specify the local iterator which is at the
        //  end of the partition data.
        static local_raw_iterator end(local_segment_iterator const& seg_iter)
        {
            return local_raw_iterator(seg_iter->end(),
                seg_iter.base()->local_data_);
        }

        // Extract the base id for the segment referenced by the given segment
        // iterator.
        static id_type get_id(segment_iterator const& iter)
        {
            return iter->get_id();
        }
    };

    template <typename T, typename Data>
    struct segmented_iterator_traits<const_vector_iterator<T, Data> >
    {
        typedef std::true_type is_segmented_iterator;

        typedef const_vector_iterator<T, Data> iterator;
        typedef typename iterator::segment_iterator segment_iterator;
        typedef typename iterator::local_segment_iterator local_segment_iterator;
        typedef typename iterator::local_iterator local_iterator;

        typedef typename local_iterator::local_raw_iterator local_raw_iterator;

        //  Conceptually this function is supposed to denote which segment
        //  the iterator is currently pointing to (i.e. just global iterator).
        static segment_iterator segment(iterator iter)
        {
            return iter.get_data()->get_const_segment_iterator(
                iter.get_global_index());
        }

        //  This function should specify which is the current segment and
        //  the exact position to which local iterator is pointing.
        static local_iterator local(iterator const& iter)
        {
            HPX_ASSERT(iter.get_data());    // avoid dereferencing end iterator
            return iter.get_data()->get_const_local_iterator(
                iter.get_global_index());
        }

        //  Build a full iterator from the segment and local iterators
        static iterator compose(segment_iterator const& seg_iter,
            local_iterator const& local_iter)
        {
            partitioned_vector<T, Data> const* data = seg_iter.get_data();
            std::size_t index = local_iter.get_local_index();
            return iterator(data, data->get_global_index(seg_iter, index));
        }

        //  This function should specify the local iterator which is at the
        //  beginning of the partition.
        static local_iterator begin(segment_iterator seg_iter)
        {
            std::size_t offset = 0;
            if (seg_iter.is_at_end())
            {
                // return iterator to the end of last segment
                --seg_iter;
                offset = seg_iter.base()->size_;
            }

            return local_iterator(seg_iter.base()->partition_, offset,
                seg_iter.base()->local_data_);
        }

        //  This function should specify the local iterator which is at the
        //  end of the partition.
        static local_iterator end(segment_iterator seg_iter)
        {
            if (seg_iter.is_at_end())
                --seg_iter;     // return iterator to the end of last segment

            auto& base = seg_iter.base();
            return local_iterator(base->partition_, base->size_,
                base->local_data_);
        }

        //  This function should specify the local iterator which is at the
        //  beginning of the partition data.
        static local_raw_iterator begin(local_segment_iterator const& seg_iter)
        {
            return local_raw_iterator(seg_iter->cbegin(),
                seg_iter.base()->local_data_);
        }

        //  This function should specify the local iterator which is at the
        //  end of the partition data.
        static local_raw_iterator end(local_segment_iterator const& seg_iter)
        {
            return local_raw_iterator(seg_iter->cend(),
                seg_iter.base()->local_data_);
        }

        // Extract the base id for the segment referenced by the given segment
        // iterator.
        static id_type get_id(segment_iterator const& iter)
        {
            return iter->get_id();
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // Some 'remote' iterators need to be mapped before being applied to the
    // local algorithms.
    template <typename T, typename Data>
    struct segmented_local_iterator_traits<local_vector_iterator<T, Data> >
    {
        typedef std::true_type is_segmented_local_iterator;

        typedef vector_iterator<T, Data> iterator;
        typedef local_vector_iterator<T, Data> local_iterator;
        typedef typename local_iterator::local_raw_iterator local_raw_iterator;

        // Extract base iterator from local_iterator
        static local_raw_iterator local(local_iterator it)
        {
            return it.local();
        }

        // Construct remote local_iterator from local_raw_iterator
        static local_iterator remote(local_raw_iterator it)
        {
            return it.remote();
        }
    };

    template <typename T, typename Data>
    struct segmented_local_iterator_traits<
        const_local_vector_iterator<T, Data> >
    {
        typedef std::true_type is_segmented_local_iterator;

        typedef const_vector_iterator<T, Data> iterator;
        typedef const_local_vector_iterator<T, Data> local_iterator;
        typedef typename local_iterator::local_raw_iterator local_raw_iterator;

        // Extract base iterator from local_iterator
        static local_raw_iterator local(local_iterator it)
        {
            return it.local();
        }

        // Construct remote local_iterator from local_raw_iterator
        static local_iterator remote(local_raw_iterator it)
        {
            return it.remote();
        }
    };
}}

#endif //  SEGMENTED_ITERATOR_HPP
