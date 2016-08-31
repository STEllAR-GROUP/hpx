//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c)      2014 Thomas Heller
//  Copyright (c)      2015 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_OUTPUT_CONTAINER_HPP
#define HPX_SERIALIZATION_OUTPUT_CONTAINER_HPP

#include <hpx/config.hpp>
#include <hpx/lcos_fwd.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/serialization/binary_filter.hpp>
#include <hpx/runtime/serialization/container.hpp>
#include <hpx/runtime/serialization/serialization_chunk.hpp>
#include <hpx/util/assert.hpp>

#include <cstddef> // for size_t
#include <cstdint>
#include <cstring> // for memcpy
#include <vector>

namespace hpx { namespace serialization
{
    namespace detail
    {
        template <typename Container>
        struct access_data
        {
            static bool is_preprocessing() { return false; }

            static void await_future(
                Container& cont
              , hpx::lcos::detail::future_data_refcnt_base & future_data)
            {}

            static void add_gid(Container& cont,
                    naming::gid_type const & gid,
                    naming::gid_type const & splitted_gid)
            {}

            static bool has_gid(Container& cont, naming::gid_type const& gid)
            {
                return false;
            }

            static void write(Container& cont, std::size_t count,
                std::size_t current, void const* address)
            {
                if (count == 1)
                    cont[current] = *static_cast<unsigned char const*>(address);
                else
                    std::memcpy(&cont[current], address, count);
            }

            static bool flush(binary_filter* filter, Container& cont,
                std::size_t current, std::size_t size, std::size_t& written)
            {
                return filter->flush(&cont[current], size, written);
            }

            static void reset(Container& cont)
            {}
        };
    }

    template <typename Container>
    struct output_container : erased_output_container
    {
    private:
        std::size_t get_chunk_size(std::size_t chunk) const
        {
            return (*chunks_)[chunk].size_;
        }

        void set_chunk_size(std::size_t chunk, std::size_t size)
        {
            (*chunks_)[chunk].size_ = size;
        }

        std::uint8_t get_chunk_type(std::size_t chunk) const
        {
            return (*chunks_)[chunk].type_;
        }

        chunk_data get_chunk_data(std::size_t chunk) const
        {
            return (*chunks_)[chunk].data_;
        }

        std::size_t get_num_chunks() const
        {
            return chunks_->size();
        }

    public:
        output_container(Container& cont,
            std::vector<serialization_chunk>* chunks,
            binary_filter* filter)
            : cont_(cont), current_(0), start_compressing_at_(0), filter_(nullptr),
              chunks_(chunks), current_chunk_(std::size_t(-1))
        {
            if (chunks_)
            {
                chunks_->push_back(create_index_chunk(0, 0));
                current_chunk_ = 0;
            }
        }

        ~output_container()
        {
            if (filter_) {
                std::size_t written = 0;

                if (cont_.size() < current_)
                    cont_.resize(current_);
                current_ = start_compressing_at_;

                do {
                    bool flushed = detail::access_data<Container>::flush(
                        filter_, cont_, current_, cont_.size()-current_, written);

                    current_ += written;
                    if (flushed)
                        break;

                    // resize container
                    cont_.resize(cont_.size()*2);

                } while (true);

                cont_.resize(current_);         // truncate container
            }
            else if (chunks_) {
                HPX_ASSERT(get_num_chunks() > current_chunk_);
                HPX_ASSERT(
                    get_chunk_type(current_chunk_) == chunk_type_index ||
                    get_chunk_size(current_chunk_) != 0);

                // complement current serialization_chunk by setting its length
                if (get_chunk_type(current_chunk_) == chunk_type_index)
                {
                    HPX_ASSERT(get_chunk_size(current_chunk_) == 0);

                    set_chunk_size(current_chunk_,
                        current_ - get_chunk_data(current_chunk_).index_);
                }
            }
        }

        bool is_preprocessing() const
        {
            return detail::access_data<Container>::is_preprocessing();
        }

        void await_future(
            hpx::lcos::detail::future_data_refcnt_base & future_data)
        {
            detail::access_data<Container>::await_future(cont_, future_data);
        }

        void add_gid(
            naming::gid_type const & gid,
            naming::gid_type const & splitted_gid)
        {
            detail::access_data<Container>::add_gid(cont_, gid, splitted_gid);
        }

        bool has_gid(naming::gid_type const & gid)
        {
            return detail::access_data<Container>::has_gid(cont_, gid);
        }

        void reset()
        {
            return detail::access_data<Container>::reset(cont_);
        }

        void set_filter(binary_filter* filter) // override
        {
            HPX_ASSERT(nullptr == filter_);
            filter_ = filter;
            start_compressing_at_ = current_;

            if (chunks_) {
                HPX_ASSERT(get_num_chunks() == 1 && get_chunk_size(0) == 0);
                chunks_->clear();
            }
        }

        void save_binary(void const* address, std::size_t count) // override
        {
            HPX_ASSERT(count != 0);
            {
                if (filter_) {
                    filter_->save(address, count);
                }
                else {
                    // make sure there is a current serialization_chunk descriptor
                    // available
                    if (chunks_)
                    {
                        HPX_ASSERT(get_num_chunks() > current_chunk_);
                        if (get_chunk_type(current_chunk_) == chunk_type_pointer ||
                            get_chunk_size(current_chunk_) != 0)
                        {
                            // add a new serialization_chunk,
                            // the chunk size will be set at the end
                            chunks_->push_back(
                                create_index_chunk(current_, 0));
                            ++current_chunk_;
                        }
                    }

                    if (cont_.size() < current_ + count)
                        cont_.resize(cont_.size() + count);

                    detail::access_data<Container>::write(
                        cont_, count, current_, address);
                }
                current_ += count;
            }
        }

        void save_binary_chunk(void const* address, std::size_t count) // override
        {
            if (filter_ || chunks_ == nullptr ||
                count < HPX_ZERO_COPY_SERIALIZATION_THRESHOLD)
            {
                // fall back to serialization_chunk-less archive
                this->output_container::save_binary(address, count);
            }
            else {
                HPX_ASSERT(get_num_chunks() > current_chunk_);
                HPX_ASSERT(
                    get_chunk_type(current_chunk_) == chunk_type_index ||
                    get_chunk_size(current_chunk_) != 0);

                // complement current serialization_chunk by setting its length
                if (get_chunk_type(current_chunk_) == chunk_type_index)
                {
                    HPX_ASSERT(get_chunk_size(current_chunk_) == 0);

                    set_chunk_size(current_chunk_,
                        current_ - get_chunk_data(current_chunk_).index_);
                }

                // add a new serialization_chunk referring to the external
                // buffer
                chunks_->push_back(create_pointer_chunk(address, count));
                ++current_chunk_;
            }
        }

        Container& cont_;
        std::size_t current_;
        std::size_t start_compressing_at_;
        binary_filter* filter_;

        std::vector<serialization_chunk>* chunks_;
        std::size_t current_chunk_;
    };
}}

#endif
