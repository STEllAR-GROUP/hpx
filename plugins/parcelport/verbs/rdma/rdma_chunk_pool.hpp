// Copyright (C) 2016 John Biddiscombe
// Copyright (C) 2000, 2001 Stephen Cleary
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at

// This is a cut down version of boost pool, adapted to handled pinned memory blocks
// we are not concerned with high performance during malloc because we use this pool
// as a simple way of pinning a large block and dividing it up into smaller blocks
// which will then be stored in a threadsafe lockfree::stack so that they can be
// quickly popped and pushed when needed or freed.

#ifndef HPX_PARCELSET_POLICIES_VERBS_PINNED_MEMORY_POOL
#define HPX_PARCELSET_POLICIES_VERBS_PINNED_MEMORY_POOL

#define BOOST_POOL_INSTRUMENT 1

#include <hpx/config.hpp>
#include "memory_region.hpp"

// std::less, std::less_equal, std::greater
#include <functional>
// new[], delete[], std::nothrow
#include <new>
// std::size_t, std::ptrdiff_t
#include <cstddef>
// std::malloc, std::free
#include <cstdlib>
// std::invalid_argument
#include <exception>
// std::max
#include <algorithm>

#include <boost/pool/poolfwd.hpp>

// boost::boost::integer::static_lcm
#include <boost/integer/common_factor_ct.hpp>
// boost::simple_segregated_storage
#include <boost/pool/simple_segregated_storage.hpp>
// boost::alignment_of
#include <boost/type_traits/alignment_of.hpp>
// BOOST_ASSERT
#include <boost/assert.hpp>

#ifdef BOOST_POOL_INSTRUMENT
#include <iostream>
#include <iomanip>
#endif

// There are a few places in this file where the expression "this->m" is used.
// This expression is used to force instantiation-time name lookup, which I am
//   informed is required for strict Standard compliance.  It's only necessary
//   if "m" is a member of a base class that is dependent on a template
//   parameter.
// Thanks to Jens Maurer for pointing this out!

/*!
  \file
  \brief Provides class \ref pool: a fast memory allocator that guarantees proper alignment of all allocated chunks,
  and which extends and generalizes the framework provided by the simple segregated storage solution.
  Also provides two UserAllocator classes which can be used in conjuction with \ref pool.
 */

/*!
  \mainpage Boost.Pool Memory Allocation Scheme
  \section intro_sec Introduction
   Pool allocation is a memory allocation scheme that is very fast, but limited in its usage.
   This Doxygen-style documentation is complementary to the
   full Quickbook-generated html and pdf documentation at www.boost.org.
  This page generated from file pool.hpp.
 */

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable:4127)  // Conditional expression is constant
#endif


namespace hpx {
namespace parcelset {
namespace policies {
namespace verbs {

    // --------------------------------------------------------------------
    // allocate an rdma_memory_region and register the memory
    struct memory_region_allocator
    {
        typedef std::size_t    size_type;
        typedef std::ptrdiff_t difference_type;

        memory_region_allocator() {}

        static rdma_memory_region_ptr malloc(rdma_protection_domain_ptr pd, const size_type bytes) {
            rdma_memory_region_ptr region = std::make_shared<rdma_memory_region>();
            LOG_DEBUG_MSG("Allocating " << decnumber(bytes) << "using chunk mallocator");
            region->allocate(pd, bytes);
            return region;
        }

        static void free(rdma_memory_region_ptr region) {
            LOG_DEBUG_MSG("Freeing a block from chunk mallocator (ref count) "
                << region.use_count());
            region.reset();
        }
    };

namespace details
{  //! Implemention only.

    template<typename SizeType>
    class PODptr
    {

        /*! \details A PODptr holds the location and size of a memory block allocated from the system.
             Each memory block is split logically into three sections:
             <b>Chunk area</b>. This section may be different sizes.
             PODptr does not care what the size of the chunks is,
             but it does care (and keep track of) the total size of the chunk area.
             <b>Next pointer</b>. This section is always the same size for a
             given SizeType. It holds a pointer
             to the location of the next memory block in the memory block list,
             or 0 if there is no such block.
             <b>Next size</b>. This section is always the same size for a
             given SizeType. It holds the size of the
             next memory block in the memory block list.
             The PODptr class just provides cleaner ways of dealing with raw
             memory blocks.
             A PODptr object is either valid or invalid.
             An invalid PODptr is analogous to a null pointer.
             The default constructor for PODptr will result in an invalid object.
             Calling the member function invalidate will result in that object
             becoming invalid.
             The member function valid can be used to test for validity.
         */
    public:
        typedef SizeType size_type;

    private:
        char * ptr;
        size_type sz;

        char * ptr_next_size() const
        {
            return (ptr + sz - sizeof(size_type));
        }
        char * ptr_next_ptr() const
        {
            return (ptr_next_size() -
                boost::integer::static_lcm<sizeof(size_type), sizeof(void *)>::value);
        }

    public:
        PODptr(char * const nptr, const size_type nsize)
            : ptr(nptr), sz(nsize)
        {
            //! A PODptr may be created to point to a memory block by passing
            //! the address and size of that memory block into the constructor.
            //! A PODptr constructed in this way is valid.
        }

        PODptr()
        :
            ptr(0), sz(0)
        { //! default constructor for PODptr will result in an invalid object.
        }

        bool valid() const
        { //! A PODptr object is either valid or invalid.
            //! An invalid PODptr is analogous to a null pointer.
            //! \returns true if PODptr is valid, false if invalid.
            return (begin() != 0);
        }

        void invalidate()
        { //! Make object invalid.
            begin() = 0;
        }

        char * & begin()
        { //! Each PODptr keeps the address and size of its memory block.
            //! \returns The address of its memory block.
            return ptr;
        }

        char * begin() const
        { //! Each PODptr keeps the address and size of its memory block.
            //! \return The address of its memory block.
            return ptr;
        }

        char * end() const
        { //! \returns begin() plus element_size (a 'past the end' value).
            return ptr_next_ptr();
        }

        size_type total_size() const
        { //! Each PODptr keeps the address and size of its memory block.
            //! The address may be read or written by the member functions begin.
            //! The size of the memory block may only be read,
            //! \returns size of the memory block.
            return sz;
        }

        size_type element_size() const
        { //! \returns size of element pointer area.
            return static_cast<size_type>(sz - sizeof(size_type) -
                boost::integer::static_lcm<sizeof(size_type), sizeof(void *)>::value);
        }

        size_type & next_size() const
        { //!
            //! \returns next_size.
            return *(static_cast<size_type *>(static_cast<void*>((ptr_next_size()))));
        }

        char * & next_ptr() const
        {  //! \returns pointer to next pointer area.
            return *(static_cast<char **>(static_cast<void*>(ptr_next_ptr())));
        }

        PODptr next() const
        { //! \returns next PODptr.
            return PODptr<size_type>(next_ptr(), next_size());
        }

        void next(const PODptr & arg) const
        { //! Sets next PODptr.
            next_ptr() = arg.begin();
            next_size() = arg.total_size();
        }
    };
    // class PODptr
} // namespace details

    /*!
  \brief A fast memory allocator that guarantees proper alignment of all allocated chunks.
  \details Whenever an object of type pool needs memory from the system,
  it will request it from its UserAllocator template parameter.
  The amount requested is determined using a doubling algorithm;
  that is, each time more system memory is allocated,
  the amount of system memory requested is doubled.
  Users may control the doubling algorithm by using the following extensions:
  Users may pass an additional constructor parameter to pool.
  This parameter is of type size_type,
  and is the number of chunks to request from the system
  the first time that object needs to allocate system memory.
  The default is 32. This parameter may not be 0.
  Users may also pass an optional third parameter to pool's
  constructor.  This parameter is of type size_type,
  and sets a maximum size for allocated chunks.  When this
  parameter takes the default value of 0, then there is no upper
  limit on chunk size.
  Finally, if the doubling algorithm results in no memory
  being allocated, the pool will backtrack just once, halving
  the chunk size and trying again.
  <b>UserAllocator type</b> - the method that the Pool will use to allocate memory from the system.
  There are essentially two ways to use class pool: the client can call \ref malloc() and \ref free() to allocate
  and free single chunks of memory, this is the most efficient way to use a pool, but does not allow for
  the efficient allocation of arrays of chunks.  Alternatively, the client may call \ref ordered_malloc() and \ref
  ordered_free(), in which case the free list is maintained in an ordered state, and efficient allocation of arrays
  of chunks are possible.  However, this latter option can suffer from poor performance when large numbers of
  allocations are performed.
     */
    template <typename UserAllocator>
    class rdma_chunk_pool:
        protected boost::simple_segregated_storage<typename UserAllocator::size_type>
    {
    public:
        typedef UserAllocator user_allocator; //!< User allocator.
        typedef typename UserAllocator::size_type size_type;  //!< An unsigned integral type that can represent the size of the largest object to be allocated.
        typedef typename UserAllocator::difference_type difference_type;  //!< A signed integral type that can represent the difference of any two pointers.

    private:
        BOOST_STATIC_CONSTANT(size_type, min_alloc_size =
            (::boost::integer::static_lcm<sizeof(void *), sizeof(size_type)>::value) );

        BOOST_STATIC_CONSTANT(size_type, min_align =
            (::boost::integer::static_lcm< ::boost::alignment_of<void *>::value,
                ::boost::alignment_of<size_type>::value>::value) );

        //! \returns 0 if out-of-memory.
        //! Called if malloc needs to resize the free list.
        void* malloc_need_resize();

    protected:
        details::PODptr<size_type> list; //!< List structure holding ordered blocks.

        boost::simple_segregated_storage<size_type> & store()
        { //! \returns pointer to store.
            return *this;
        }
        const boost::simple_segregated_storage<size_type> & store() const
        { //! \returns pointer to store.
            return *this;
        }

        const size_type requested_size;
        size_type next_size;
        size_type start_size;
        size_type max_size;
        rdma_protection_domain_ptr pd_;

        // each time we allocate a new block, we store the rdma region data
        std::unordered_map<char *, rdma_memory_region_ptr> region_map;

        //! finds which POD in the list 'chunk' was allocated from.
        details::PODptr<size_type> find_POD(void * const chunk) const;

        // is_from() tests a chunk to determine if it belongs in a block.
        static bool is_from(void * const chunk, char * const i,
            const size_type sizeof_i)
        { //! \param chunk chunk to check if is from this pool.
            //! \param i memory chunk at i with element sizeof_i.
            //! \param sizeof_i element size (size of the chunk area of that block, not the total size of that block).
            //! \returns true if chunk was allocated or may be returned.
            //! as the result of a future allocation.
            //!
            //! Returns false if chunk was allocated from some other pool,
            //! or may be returned as the result of a future allocation from some other pool.
            //! Otherwise, the return value is meaningless.
            //!
            //! Note that this function may not be used to reliably test random pointer values.

            // We use std::less_equal and std::less to test 'chunk'
            //  against the array bounds because standard operators
            //  may return unspecified results.
            // This is to ensure portability.  The operators < <= > >= are only
            //  defined for pointers to objects that are 1) in the same array, or
            //  2) subobjects of the same object [5.9/2].
            // The functor objects guarantee a total order for any pointer [20.3.3/8]
            std::less_equal<void *> lt_eq;
            std::less<void *> lt;
            return (lt_eq(i, chunk) && lt(chunk, i + sizeof_i));
        }

        size_type alloc_size() const
        { //!  Calculated size of the memory chunks that will be allocated by this Pool.
            //! \returns allocated size.
            // For alignment reasons, this used to be defined to be lcm(requested_size, sizeof(void *), sizeof(size_type)),
            // but is now more parsimonious: just rounding up to the minimum required alignment of our housekeeping data
            // when required.  This works provided all alignments are powers of two.
            size_type s = (std::max)(requested_size, min_alloc_size);
            size_type rem = s % min_align;
            if(rem)
                s += min_align - rem;
            BOOST_ASSERT(s >= min_alloc_size);
            BOOST_ASSERT(s % min_align == 0);
            return s;
        }

        static void * & nextof(void * const ptr)
        { //! \returns Pointer dereferenced.
            //! (Provided and used for the sake of code readability :)
            return *(static_cast<void **>(ptr));
        }

    public:
        // pre: npartition_size != 0 && nnext_size != 0
        explicit rdma_chunk_pool(
            rdma_protection_domain_ptr pd,
            const size_type nrequested_size,
            const size_type nnext_size = 32,
            const size_type nmax_size = 0)
        :
            list(0, 0), requested_size(nrequested_size), next_size(nnext_size),
            start_size(nnext_size), max_size(nmax_size), pd_(pd)
        { //!   Constructs a new empty Pool that can be used to allocate chunks of size RequestedSize.
            //! \param nrequested_size  Requested chunk size
            //! \param  nnext_size parameter is of type size_type,
            //!   is the number of chunks to request from the system
            //!   the first time that object needs to allocate system memory.
            //!   The default is 32. This parameter may not be 0.
            //! \param nmax_size is the maximum number of chunks to allocate in one block.
        }

        ~rdma_chunk_pool()
        { //!   Destructs the Pool, freeing its list of memory blocks.
            purge_memory();
        }

        // Releases memory blocks that don't have chunks allocated
        // pre: lists are ordered
        //  Returns true if memory was actually deallocated
        bool release_memory();

        // Releases *all* memory blocks, even if chunks are still allocated
        //  Returns true if memory was actually deallocated
        bool purge_memory();

        size_type get_next_size() const
        { //! Number of chunks to request from the system the next time that object needs to allocate system memory. This value should never be 0.
            //! \returns next_size;
            return next_size;
        }
        void set_next_size(const size_type nnext_size)
        { //! Set number of chunks to request from the system the next time that object needs to allocate system memory. This value should never be set to 0.
            //! \returns nnext_size.
            next_size = start_size = nnext_size;
        }
        size_type get_max_size() const
        { //! \returns max_size.
            return max_size;
        }
        void set_max_size(const size_type nmax_size)
        { //! Set max_size.
            max_size = nmax_size;
        }
        size_type get_requested_size() const
        { //!   \returns the requested size passed into the constructor.
            //! (This value will not change during the lifetime of a Pool object).
            return requested_size;
        }

        // Both malloc and ordered_malloc do a quick inlined check first for any
        // free chunks.  Only if we need to get another memory block do we call
        // the non-inlined *_need_resize() functions.
        // Returns 0 if out-of-memory
        rdma_memory_region malloc BOOST_PREVENT_MACRO_SUBSTITUTION()
        { //! Allocates a chunk of memory. Searches in the list of memory blocks
            //! for a block that has a free chunk, and returns that free chunk if found.
            //! Otherwise, creates a new memory block, adds its free list to pool's free list,
            //! \returns a free chunk from that block.
            //! If a new memory block cannot be allocated, returns 0. Amortized O(1).
            // Look for a non-empty storage

            // when a block is taken from the underlying segregated store,
            // we find the POD chunk it came from and assign the rdma_region
            // to it from that chunk.

            void *data_chunk;
            if (!store().empty()) {
                data_chunk = (store().malloc)();
            }
            else {
                data_chunk = (malloc_need_resize)();
            }
            auto pod = find_POD(data_chunk);
            std::ptrdiff_t offset = (static_cast<char*>(data_chunk) - pod.begin());
            //
            struct ibv_mr *region = region_map[pod.begin()]->get_region();
            rdma_memory_region chunk(
                region,
                static_cast<char*>(region->addr) + offset,
                rdma_memory_region::BLOCK_PARTIAL,
                requested_size
            );
            //
            return chunk;
        }

        // pre: 'chunk' must have been previously
        //        returned by *this.malloc().
        void free BOOST_PREVENT_MACRO_SUBSTITUTION(rdma_memory_region chunk)
        {
            if (!chunk.get_partial_region()) {
                LOG_ERROR_MSG("Chunk was not allocated from this pool correctly");
                throw std::runtime_error("Chunk was not allocated from this pool correctly");
            }
            (store().free)(chunk.get_address());
        }

        // is_from() tests a chunk to determine if it was allocated from *this
        bool is_from(void * const chunk) const
        { //! \returns Returns true if chunk was allocated from u or
            //! may be returned as the result of a future allocation from u.
            //! Returns false if chunk was allocated from some other pool or
            //! may be returned as the result of a future allocation from some other pool.
            //! Otherwise, the return value is meaningless.
            //! Note that this function may not be used to reliably test random pointer values.
            return (find_POD(chunk).valid());
        }
    };

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
    template <typename UserAllocator>
    typename rdma_chunk_pool<UserAllocator>::size_type const rdma_chunk_pool<UserAllocator>::min_alloc_size;
    template <typename UserAllocator>
    typename rdma_chunk_pool<UserAllocator>::size_type const rdma_chunk_pool<UserAllocator>::min_align;
#endif

    template <typename UserAllocator>
    bool rdma_chunk_pool<UserAllocator>::release_memory()
    { //! pool must be ordered. Frees every memory block that doesn't have any allocated chunks.
        //! \returns true if at least one memory block was freed.

        // ret is the return value: it will be set to true when we actually call
        //  UserAllocator::free(..)
        bool ret = false;

        // This is a current & previous iterator pair over the memory block list
        details::PODptr<size_type> ptr = list;
        details::PODptr<size_type> prev;

        // This is a current & previous iterator pair over the free memory chunk list
        //  Note that "prev_free" in this case does NOT point to the previous memory
        //  chunk in the free list, but rather the last free memory chunk before the
        //  current block.
        void * free_p = this->first;
        void * prev_free_p = 0;

        const size_type partition_size = alloc_size();

        // Search through all the all the allocated memory blocks
        while (ptr.valid())
        {
            // At this point:
            //  ptr points to a valid memory block
            //  free_p points to either:
            //    0 if there are no more free chunks
            //    the first free chunk in this or some next memory block
            //  prev_free_p points to either:
            //    the last free chunk in some previous memory block
            //    0 if there is no such free chunk
            //  prev is either:
            //    the PODptr whose next() is ptr
            //    !valid() if there is no such PODptr

            // If there are no more free memory chunks, then every remaining
            //  block is allocated out to its fullest capacity, and we can't
            //  release any more memory
            if (free_p == 0)
                break;

            // We have to check all the chunks.  If they are *all* free (i.e., present
            //  in the free list), then we can free the block.
            bool all_chunks_free = true;

            // Iterate 'i' through all chunks in the memory block
            // if free starts in the memory block, be careful to keep it there
            void * saved_free = free_p;
            for (char * i = ptr.begin(); i != ptr.end(); i += partition_size)
            {
                // If this chunk is not free
                if (i != free_p)
                {
                    // We won't be able to free this block
                    all_chunks_free = false;

                    // free_p might have travelled outside ptr
                    free_p = saved_free;
                    // Abort searching the chunks; we won't be able to free this
                    //  block because a chunk is not free.
                    break;
                }

                // We do not increment prev_free_p because we are in the same block
                free_p = nextof(free_p);
            }

            // post: if the memory block has any chunks, free_p points to one of them
            // otherwise, our assertions above are still valid

            const details::PODptr<size_type> next = ptr.next();

            if (!all_chunks_free)
            {
                if (is_from(free_p, ptr.begin(), ptr.element_size()))
                {
                    std::less<void *> lt;
                    void * const end = ptr.end();
                    do
                    {
                        prev_free_p = free_p;
                        free_p = nextof(free_p);
                    } while (free_p && lt(free_p, end));
                }
                // This invariant is now restored:
                //     free_p points to the first free chunk in some next memory block, or
                //       0 if there is no such chunk.
                //     prev_free_p points to the last free chunk in this memory block.

                // We are just about to advance ptr.  Maintain the invariant:
                // prev is the PODptr whose next() is ptr, or !valid()
                // if there is no such PODptr
                prev = ptr;
            }
            else
            {
                // All chunks from this block are free

                // Remove block from list
                if (prev.valid())
                    prev.next(next);
                else
                    list = next;

                // Remove all entries in the free list from this block
                if (prev_free_p != 0)
                    nextof(prev_free_p) = free_p;
                else
                    this->first = free_p;

                // delete the storage, and release memory region
                char *base_ptr = find_POD(ptr.begin()).begin();
                rdma_memory_region_ptr region = region_map[base_ptr];
                region_map.erase(base_ptr);
                (UserAllocator::free)(region);

                ret = true;
            }

            // Increment ptr
            ptr = next;
        }

        next_size = start_size;
        return ret;
    }

    template <typename UserAllocator>
    bool rdma_chunk_pool<UserAllocator>::purge_memory()
    { //! pool must be ordered.
        //! Frees every memory block.
        //!
        //! This function invalidates any pointers previously returned
        //! by allocation functions of t.
        //! \returns true if at least one memory block was freed.

        details::PODptr<size_type> iter = list;

        if (!iter.valid())
            return false;

        do
        {
            // hold "next" pointer
            const details::PODptr<size_type> next = iter.next();

            // delete the storage, and release memory region
            char *base_ptr = find_POD(iter.begin()).begin();
            rdma_memory_region_ptr region = region_map[base_ptr];
            region_map.erase(base_ptr);
            (UserAllocator::free)(region);

            // increment iter
            iter = next;
        } while (iter.valid());

        list.invalidate();
        this->first = 0;
        next_size = start_size;

        return true;
    }

    template <typename UserAllocator>
    void *rdma_chunk_pool<UserAllocator>::malloc_need_resize()
    { //! No memory in any of our storages; make a new storage,
        //!  Allocates chunk in newly malloc after resize.
        //! \returns pointer to chunk.
        size_type partition_size = alloc_size();
        size_type POD_size = static_cast<size_type>(next_size * partition_size +
            boost::integer::static_lcm<sizeof(size_type), sizeof(void *)>::value + sizeof(size_type));

        rdma_memory_region_ptr ptr = (UserAllocator::malloc)(pd_, POD_size);
        if (ptr == 0)
        {
            if (next_size > 4)
            {
                next_size >>= 1;
                partition_size = alloc_size();
                POD_size = static_cast<size_type>(next_size * partition_size +
                    boost::integer::static_lcm<sizeof(size_type), sizeof(void *)>::value + sizeof(size_type));
                ptr = (UserAllocator::malloc)(pd_, POD_size);
            }
            if (ptr == 0)
                return 0;
        }
        const details::PODptr<size_type> node(ptr->get_base_address(), POD_size);
        region_map[ptr->get_base_address()] = ptr;

        BOOST_USING_STD_MIN();
        if(!max_size)
            next_size <<= 1;
        else if( next_size*partition_size/requested_size < max_size)
            next_size = min BOOST_PREVENT_MACRO_SUBSTITUTION(next_size << 1, max_size*requested_size/ partition_size);

        //  initialize it,
        store().add_block(node.begin(), node.element_size(), partition_size);

        //  insert it into the list,
        node.next(list);
        list = node;

        //  and return a chunk from it.
        return (store().malloc)();
    }

    template <typename UserAllocator>
    details::PODptr<typename rdma_chunk_pool<UserAllocator>::size_type>
    rdma_chunk_pool<UserAllocator>::find_POD(void * const chunk) const
    { //! find which PODptr storage memory that this chunk is from.
        //! \returns the PODptr that holds this chunk.
        // Iterate down list to find which storage this chunk is from.
        details::PODptr<size_type> iter = list;
        while (iter.valid())
        {
            if (is_from(chunk, iter.begin(), iter.element_size()))
                return iter;
            iter = iter.next();
        }

        return iter;
    }


}}}}

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif