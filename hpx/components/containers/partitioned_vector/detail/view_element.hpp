//  Copyright (c) 2017 Antoine Tran Tan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/components/partitioned_vector/partitioned_vector_view.hpp

#ifndef HPX_PARTITIONED_VECTOR_DETAIL_VIEW_ELEMENT_HPP
#define HPX_PARTITIONED_VECTOR_DETAIL_VIEW_ELEMENT_HPP

#include <hpx/components/containers/partitioned_vector/partitioned_vector_component.hpp>
#include <hpx/components/containers/partitioned_vector/partitioned_vector_segmented_iterator.hpp>
#include <hpx/lcos/spmd_block.hpp>
#include <hpx/runtime/get_locality_id.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/naming/name.hpp>

#include <utility>
#include <cstdint>
#include <cstddef>

///////////////////////////////////////////////////////////////////////////////
/// \cond NOINTERNAL

namespace hpx { namespace detail
{
    template<typename T, typename Data>
    struct view_element
    : public hpx::partitioned_vector_partition<T,Data>
    {
        using pvector_iterator = hpx::vector_iterator<T,Data>;
        using segment_iterator
            = typename pvector_iterator::segment_iterator;
        using local_segment_iterator
            = typename pvector_iterator::local_segment_iterator;
        using traits
            = typename hpx::traits::segmented_iterator_traits<pvector_iterator>;

    public:
        explicit view_element(
            hpx::lcos::spmd_block const & block,
                segment_iterator begin, segment_iterator end, segment_iterator it)

        : hpx::partitioned_vector_partition<T,Data>( it->get_id() ), it_(it)
        {
            std::uint32_t here = hpx::get_locality_id();

            is_data_here_
            = ( here == hpx::naming::get_locality_id_from_id(it_->get_id()) );

            is_owned_by_current_thread_ =
                is_data_here_ && (
                    (std::distance(
                        local_segment_iterator(begin.base(),end.base(),here)
                        ,local_segment_iterator(it_.base(),end.base(),here)
                        )  %  block.get_images_per_locality() )
                    == (block.this_image() %  block.get_images_per_locality()) );
        }

        view_element(view_element const &) = default;

        // Not copy-assygnable
        view_element& operator=(view_element const &) = delete;

        // But movable
        view_element(view_element && other) = default;

        // Explicit conversion allows to perform Get operations
        explicit operator Data() const { return const_data(); }

        // operator overloading (Useful for manual view definition)
        segment_iterator && operator&()
        {
            return std::move(it_);

        }

    private:
        bool is_data_here() const
        {
            return is_data_here_;
        }

        Data const_data() const
        {
            if ( is_data_here() )
            {
                return this->get_ptr()->get_data();
            }
            else
                return this->get_copied_data(hpx::launch::sync) ;
        }

    public:
        Data & data()
        {
            return this->get_ptr()->get_data();
        }

        Data const & data() const
        {
            return this->get_ptr()->get_data();
        }

        bool is_owned_by_current_thread() const
        {
            return is_owned_by_current_thread_;
        }

        // Note: Put operation. Race condition may occur, be sure that
        // operator=() is called by only one thread at a time.
        void operator=(Data && other)
        {
            if ( is_data_here() )
            {
                Data & ref = data();

                HPX_ASSERT_MSG( ref.size() == other.size(), \
                    "r-value vector has invalid size");

                ref = other;
            }
            else
            {
                this->set_data(hpx::launch::sync, std::move(other) );
            }
        }

        // Note: Put operation. Free of race conditions.
        void operator=(view_element<T,Data> && other)
        {
            if(other.is_owned_by_current_thread())
            {
                if( is_data_here() )
                {
                    Data & ref = data();

                    HPX_ASSERT_MSG( ref.size() == other.size(),
                        "Indexed r-value element has " \
                        "invalid size");

                    ref = other.data();
                }

                else
                    this->set_data(hpx::launch::sync, other.const_data() );
            }
        }

        T operator[](std::size_t i) const
        {
            if( is_data_here() )
            {
                return data()[i];
            }

            else
                return this->get_value(hpx::launch::sync,i);
        }

    private:
        bool is_data_here_;
        bool is_owned_by_current_thread_;
        segment_iterator it_;
    };
}}

#endif // HPX_PARTITIONED_VECTOR_DETAIL_VIEW_ELEMENT_HPP
