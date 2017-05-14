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
#include <memory>
#include <type_traits>
#include <vector>

namespace hpx { namespace serialization
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Container>
        struct access_data
        {
            typedef std::false_type preprocessing_only;

            HPX_CONSTEXPR static bool is_preprocessing() { return false; }

            HPX_CONSTEXPR static void await_future(
                Container& cont
              , hpx::lcos::detail::future_data_refcnt_base & future_data)
            {}

            HPX_CONSTEXPR static void add_gid(Container& cont,
                naming::gid_type const & gid,
                naming::gid_type const & split_gid)
            {}

            HPX_CONSTEXPR static bool has_gid(Container& cont,
                naming::gid_type const& gid)
            {
                return false;
            }

            static void write(Container& cont, std::size_t count,
                std::size_t current, void const* address)
            {
                void* dest = &cont[current];
                switch (count)
                {
                case 8:
                    *static_cast<std::uint64_t*>(dest) =
                        *static_cast<std::uint64_t const*>(address);
                    break;

                case 4:
                    *static_cast<std::uint32_t*>(dest) =
                        *static_cast<std::uint32_t const*>(address);
                    break;

                case 2:
                    *static_cast<std::uint16_t*>(dest) =
                        *static_cast<std::uint16_t const*>(address);
                    break;

                case 1:
                    *static_cast<std::uint8_t*>(dest) =
                        *static_cast<std::uint8_t const*>(address);
                    break;

                default:
                    std::memcpy(dest, address, count);
                    break;
                }
            }

            static bool flush(binary_filter* filter, Container& cont,
                std::size_t current, std::size_t size, std::size_t& written)
            {
                return filter->flush(&cont[current], size, written);
            }

            HPX_CONSTEXPR static void reset(Container& cont)
            {}
        };

        ///////////////////////////////////////////////////////////////////////
        struct basic_chunker
        {
            HPX_CONSTEXPR basic_chunker(std::vector<serialization_chunk>*) {}

            HPX_CONSTEXPR static std::size_t get_chunk_size()
            {
                return 0;
            }

            HPX_CONSTEXPR static void set_chunk_size(std::size_t)
            {
            }

            HPX_CONSTEXPR static std::uint8_t get_chunk_type()
            {
                return chunk_type_index;
            }

            HPX_CONSTEXPR static std::size_t get_chunk_data_index()
            {
                return 0;
            }

            HPX_CONSTEXPR static std::size_t get_num_chunks()
            {
                return 1;
            }

            HPX_CONSTEXPR static void push_back(serialization_chunk && chunk) {}

            HPX_CONSTEXPR static void reset() {}
        };

        struct vector_chunker
        {
            vector_chunker(std::vector<serialization_chunk>* chunks)
              : chunks_(chunks)
            {}

            std::size_t get_chunk_size() const
            {
                return chunks_->back().size_;
            }

            void set_chunk_size(std::size_t size)
            {
                chunks_->back().size_ = size;
            }

            std::uint8_t get_chunk_type() const
            {
                return chunks_->back().type_;
            }

            std::size_t get_chunk_data_index() const
            {
                return chunks_->back().data_.index_;
            }

            std::size_t get_num_chunks() const
            {
                return chunks_->size();
            }

            void push_back(serialization_chunk && chunk)
            {
                chunks_->push_back(chunk);
            }

            void reset()
            {
                chunks_->clear();
                chunks_->push_back(create_index_chunk(0, 0));
            }

            std::vector<serialization_chunk>* chunks_;
        };

        struct counting_chunker
        {
            counting_chunker(std::vector<serialization_chunk>*)
              : num_chunks_(0)
            {}

            std::size_t get_chunk_size() const
            {
                return chunk_.size_;
            }

            void set_chunk_size(std::size_t size)
            {
                chunk_.size_ = size;
            }

            std::uint8_t get_chunk_type() const
            {
                return chunk_.type_;
            }

            std::size_t get_chunk_data_index() const
            {
                return chunk_.data_.index_;
            }

            std::size_t get_num_chunks() const
            {
                return num_chunks_;
            }

            void push_back(serialization_chunk && chunk)
            {
                chunk_ = chunk;
                ++num_chunks_;
            }

            void reset()
            {
                chunk_ = create_index_chunk(0, 0);
                num_chunks_ = 1;
            }

            serialization_chunk chunk_;
            std::size_t num_chunks_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Container, typename Chunker>
    struct output_container : erased_output_container
    {
        output_container(Container& cont,
                std::vector<serialization_chunk>* chunks = nullptr)
          : cont_(cont), current_(0), chunker_(chunks)
        {
            chunker_.reset();
        }

        ~output_container()
        {}

        void flush()
        {
            HPX_ASSERT(
                chunker_.get_chunk_type() == chunk_type_index ||
                chunker_.get_chunk_size() != 0);

            // complement current serialization_chunk by setting its length
            if (chunker_.get_chunk_type() == chunk_type_index)
            {
                HPX_ASSERT(chunker_.get_chunk_size() == 0);

                chunker_.set_chunk_size(
                    current_ - chunker_.get_chunk_data_index());
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
            naming::gid_type const & split_gid)
        {
            detail::access_data<Container>::add_gid(cont_, gid, split_gid);
        }

        bool has_gid(naming::gid_type const & gid)
        {
            return detail::access_data<Container>::has_gid(cont_, gid);
        }

        std::size_t get_num_chunks() const
        {
            return chunker_.get_num_chunks();
        }

        void reset()
        {
            chunker_.reset();
            detail::access_data<Container>::reset(cont_);
        }

        void set_filter(binary_filter* filter) // override
        {
            HPX_ASSERT(chunker_.get_num_chunks() == 1 &&
                chunker_.get_chunk_size() == 0);
            chunker_.reset();
        }

        void save_binary(void const* address, std::size_t count) // override
        {
            HPX_ASSERT(count != 0);

            // make sure there is a current serialization_chunk descriptor
            // available
            if (chunker_.get_chunk_type() == chunk_type_pointer ||
                chunker_.get_chunk_size() != 0)
            {
                // add a new serialization_chunk,
                // the chunk size will be set at the end
                chunker_.push_back(create_index_chunk(current_, 0));
            }

            std::size_t new_current = current_ + count;
            if (cont_.size() < new_current)
                cont_.resize(cont_.size() + count);

            detail::access_data<Container>::write(
                cont_, count, current_, address);

            current_ = new_current;
        }

        void save_binary_chunk(void const* address, std::size_t count) // override
        {
            if (count < HPX_ZERO_COPY_SERIALIZATION_THRESHOLD)
            {
                // fall back to serialization_chunk-less archive
                this->output_container::save_binary(address, count);
            }
            else {
                HPX_ASSERT(
                    chunker_.get_chunk_type() == chunk_type_index ||
                    chunker_.get_chunk_size() != 0);

                // complement current serialization_chunk by setting its length
                if (chunker_.get_chunk_type() == chunk_type_index)
                {
                    HPX_ASSERT(chunker_.get_chunk_size() == 0);

                    chunker_.set_chunk_size(
                        current_ - chunker_.get_chunk_data_index());
                }

                // add a new serialization_chunk referring to the external
                // buffer
                chunker_.push_back(create_pointer_chunk(address, count));
            }
        }

    protected:
        Container& cont_;
        std::size_t current_;
        Chunker chunker_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Container, typename Chunker>
    struct filtered_output_container : output_container<Container, Chunker>
    {
        typedef output_container<Container, Chunker> base_type;

        filtered_output_container(Container& cont,
                std::vector<serialization_chunk>* chunks = nullptr)
          : output_container<Container, Chunker>(cont, chunks),
            start_compressing_at_(0), filter_(nullptr)
        {}

        ~filtered_output_container()
        {}

        void flush()
        {
            std::size_t written = 0;

            if (this->cont_.size() < this->current_)
                this->cont_.resize(this->current_);

            this->current_ = start_compressing_at_;

            do {
                bool flushed = detail::access_data<Container>::flush(
                    filter_, this->cont_, this->current_,
                    this->cont_.size()-this->current_, written);

                this->current_ += written;
                if (flushed)
                    break;

                // resize container
                this->cont_.resize(this->cont_.size()*2);

            } while (true);

            this->cont_.resize(this->current_);         // truncate container
        }

        void set_filter(binary_filter* filter) // override
        {
            HPX_ASSERT(nullptr == filter_ && filter != nullptr);
            filter_ = filter;
            start_compressing_at_ = this->current_;

            this->base_type::set_filter(nullptr);
        }

        void save_binary(void const* address, std::size_t count) // override
        {
            HPX_ASSERT(count != 0);

            filter_->save(address, count);
            this->current_ += count;
        }

        void save_binary_chunk(void const* address, std::size_t count) // override
        {
            if (count < HPX_ZERO_COPY_SERIALIZATION_THRESHOLD)
            {
                // fall back to serialization_chunk-less archive
                HPX_ASSERT(count != 0);
                filter_->save(address, count);
                this->current_ += count;
            }
            else {
                this->base_type::save_binary_chunk(address, count);
            }
        }

    protected:
        std::size_t start_compressing_at_;
        binary_filter* filter_;
    };
}}

#endif
