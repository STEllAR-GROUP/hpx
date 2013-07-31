//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_UTIL_OCHUNK_MANAGER_JUL_31_2013_1158AM)
#define HPX_UTIL_OCHUNK_MANAGER_JUL_31_2013_1158AM

#include <hpx/util/binary_filter.hpp>

namespace hpx { namespace util { namespace detail
{
    ///////////////////////////////////////////////////////////////////////
    union chunk_data
    {
        std::size_t index_;     // position inside the data buffer
        void* pos_;             // pointer to external data buffer
    };

    enum chunk_type
    {
        chunk_type_index = 0,
        chunk_type_pointer = 1
    };

    struct chunk
    {
        chunk_type type_;
        chunk_data data_;       // index or position
        std::size_t size_;      // size of the chunk starting at pos_
    };

    ///////////////////////////////////////////////////////////////////////
    inline chunk create_index_chunk(std::size_t index, std::size_t size)
    {
        chunk retval = { chunk_type_index, 0, size };
        retval.data_.index_ = index;
        return retval;
    }

    inline chunk create_index_chunk(unsigned char* pos, std::size_t size)
    {
        chunk retval = { chunk_type_pointer, 0, size };
        retval.data_.pos_ = pos;
        return retval;
    }

    ///////////////////////////////////////////////////////////////////////
    struct erase_container_type
    {
        virtual ~erase_container_type() {}
        virtual void set_filter(binary_filter* filter) = 0;
        virtual void save_binary(void const* address, std::size_t count) = 0;
        virtual void save_binary_chunk(void const* address, std::size_t count) = 0;
    };

    template <typename Container>
    struct container_type : erase_container_type
    {
        container_type(Container& cont)
            : cont_(cont), current_(0), start_compressing_at_(0), filter_(0)
        {
            chunks_.push_back(create_index_chunk(0, 0));
            current_chunk_ = 0;
        }

        ~container_type()
        {
            if (filter_) {
                std::size_t written = 0;

                if (cont_.size() < current_)
                    cont_.resize(current_);
                current_ = start_compressing_at_;

                do {
                    bool flushed = filter_->flush(&cont_[current_],
                        cont_.size()-current_, written);

                    current_ += written;
                    if (flushed)
                        break;

                    // resize container
                    cont_.resize(cont_.size()*2);

                } while (true);

                cont_.resize(current_);         // truncate container
            }
        }

        void set_filter(binary_filter* filter)
        {
            BOOST_ASSERT(0 == filter_);
            filter_ = filter;
            start_compressing_at_ = current_;
        }

        void save_binary(void const* address, std::size_t count)
        {
            BOOST_ASSERT(count != 0);
            {
                if (filter_) {
                    filter_->save(address, count);
                }
                else {
                    // make sure there is current chunk descriptor available
                    if (chunks_[current_chunk_].size_ != 0)
                    {
                        // add a new chunk
                        chunks_.push_back(create_index_chunk(current_, 0));
                        ++current_chunk_;
                    }

                    if (cont_.size() < current_ + count)
                        cont_.resize(cont_.size() + count);

                    if (count == 1)
                        cont_[current_] = *static_cast<unsigned char const*>(address);
                    else
                        std::memcpy(&cont_[current_], address, count);
                }
                current_ += count;
            }
        }

        void save_binary_chunk(void const* address, std::size_t count)
        {
            if (filter_) {
                // fall back to chunk-less archive
                save_binary(address, count);
            }
            else {
                // complement current chunk by setting its length
                BOOST_ASSERT(chunks_[current_chunk_].type_ == chunk_type_index);
                BOOST_ASSERT(chunks_[current_chunk_].size_ == 0);

                chunks_[current_chunk_].size_ = current_ - chunks_[current_chunk_].index_;

                // add a new chunk referring to the external buffer
                chunks_.push_back(create_pointer_chunk(address, count));
                ++current_chunk_;
            }
        }

        Container& cont_;
        std::size_t current_;
        std::size_t start_compressing_at_;
        binary_filter* filter_;

        std::vector<chunk> chunks_;
        std::size_t current_chunk_;
    };
}}}

#endif

