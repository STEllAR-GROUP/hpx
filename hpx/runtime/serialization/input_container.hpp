//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c)      2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_INPUT_CONTAINER_HPP
#define HPX_SERIALIZATION_INPUT_CONTAINER_HPP

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/binary_filter.hpp>
#include <hpx/runtime/serialization/container.hpp>
#include <hpx/runtime/serialization/serialization_chunk.hpp>
#include <hpx/throw_exception.hpp>
#include <hpx/util/assert.hpp>

#include <cstddef> // for size_t
#include <cstring> // for memcpy
#include <memory>
#include <vector>

namespace hpx { namespace serialization
{
    template <typename Container>
    struct input_container: erased_input_container
    {
    private:
        std::size_t get_chunk_size(std::size_t chunk) const
        {
            return (*chunks_)[chunk].size_;
        }

        boost::uint8_t get_chunk_type(std::size_t chunk) const
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
        input_container(Container const& cont, std::size_t inbound_data_size)
          : cont_(cont), current_(0), filter_(),
            decompressed_size_(inbound_data_size),
            chunks_(0), current_chunk_(std::size_t(-1)), current_chunk_size_(0)
        {}

        input_container(Container const& cont,
                std::vector<serialization_chunk> const* chunks,
                std::size_t inbound_data_size)
          : cont_(cont), current_(0), filter_(),
            decompressed_size_(inbound_data_size),
            chunks_(0), current_chunk_(std::size_t(-1)), current_chunk_size_(0)
        {
            if (chunks && chunks->size() != 0)
            {
                chunks_ = chunks;
                current_chunk_ = 0;
            }
        }

        void set_filter(binary_filter* filter) // override
        {
            filter_.reset(filter);
            if (filter) {
                current_ = filter->init_data(&cont_[current_],
                    cont_.size()-current_, decompressed_size_);

                if (decompressed_size_ < current_)
                {
                    HPX_THROW_EXCEPTION(serialization_error
                      , "input_container::set_filter"
                      , "archive data bstream is too short");
                    return;
                }
            }
        }

        void load_binary(void* address, std::size_t count)
        {
            if (filter_) {
                filter_->load(address, count);
            }
            else {
                if (current_+count > cont_.size())
                {
                    HPX_THROW_EXCEPTION(serialization_error
                      , "input_container::load_binary"
                      , "archive data bstream is too short");
                    return;
                }

                if (count == 1)
                    *static_cast<unsigned char*>(address) = cont_[current_];
                else
                    std::memcpy(address, &cont_[current_], count);
                current_ += count;

                if (chunks_) {
                    current_chunk_size_ += count;
                    // make sure we switch to the next serialization_chunk if necessary
                    std::size_t current_chunk_size = get_chunk_size(current_chunk_);
                    if (current_chunk_size != 0 && current_chunk_size_ >=
                        current_chunk_size)
                    {
                        // raise an error if we read past the serialization_chunk
                        if (current_chunk_size_ > current_chunk_size)
                        {
                            HPX_THROW_EXCEPTION(serialization_error
                              , "input_container::load_binary"
                              , "archive data bstream structure mismatch");
                            return;
                        }
                        ++current_chunk_;
                        current_chunk_size_ = 0;
                    }
                }
            }
        }

        void load_binary_chunk(void* address, std::size_t count) // override
        {
            HPX_ASSERT((boost::int64_t)count >= 0);

            if (filter_.get() || chunks_ == 0 ||
                count < HPX_ZERO_COPY_SERIALIZATION_THRESHOLD) {
                // fall back to serialization_chunk-less archive
                this->input_container::load_binary(address, count);
            }
            else {
                HPX_ASSERT(current_chunk_ != std::size_t(-1));
                HPX_ASSERT(get_chunk_type(current_chunk_) == chunk_type_pointer);

                if (get_chunk_size(current_chunk_) != count)
                {
                    HPX_THROW_EXCEPTION(serialization_error
                      , "input_container::load_binary_chunk"
                      , "archive data bstream data chunk size mismatch");
                    return;
                }

                // unfortunately we can't implement a zero copy policy on
                // the receiving end
                // as the memory was already allocated by the serialization code
                std::memcpy(address, get_chunk_data(current_chunk_).pos_, count);
                ++current_chunk_;
            }
        }

        Container const& cont_;
        std::size_t current_;
        std::unique_ptr<binary_filter> filter_;
        std::size_t decompressed_size_;

        std::vector<serialization_chunk> const* chunks_;
        std::size_t current_chunk_;
        std::size_t current_chunk_size_;
    };
}}

#endif
