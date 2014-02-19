// Copyright (c) 2010 Maciej Brodowicz
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_3ED714E9_F359_423C_93DC_B819A313BBAF)
#define HPX_3ED714E9_F359_423C_93DC_B819A313BBAF

#include "pci.hh"

#include <hpx/config.hpp>
#include <hpx/util/hardware/queue_config.hpp>

#include <boost/type_traits/is_pod.hpp>
#include <boost/cstdint.hpp>

namespace hpx { namespace util { namespace hardware
{

// Wrapper for hardware FPGA fifo. Operates on PODs up to 8 bytes in size.
template<typename T>
class fifo
{
    BOOST_STATIC_ASSERT(boost::is_pod<T>::value && sizeof(T) <= 8);

    enum
    {
        XilinxID = 0x10ee // Xilinx vendor ID
    };

    // pointer to PCI device descriptor
    pci::Device *dev_;
    // base address of the first PCI memory space mapped into application
    // address space
    char *base_;
    boost::uint64_t tnum_;

  protected:
    // hw request to address mapping;
    // requires OS thread number (small integer) to select hardware
    // state necessary to manage split PCI requests
    boost::uint64_t *cmd2addr(unsigned cmd)
    {
        return reinterpret_cast<boost::uint64_t*>
            (base_+((cmd|(tnum_<<TM_REQ_BITS))<<3));
    }

    // find the device and map the first available memory BAR
    void map_memory()
    {
        if (pci_system_init() != 0)
            throw std::runtime_error("PCI system initialization failed");

        // match any Xilinx board
        // (temporary, as there are also boards from Avnet and HTG)
        dev_ = new pci::Device(XilinxID, PCI_MATCH_ANY);

        if (!dev_ || !dev_->valid())
            throw std::runtime_error("Failed to find the PCI FPGA board");

        base_ = reinterpret_cast<char *>(dev_->region(PCI_BAR_ANY));

        if (!base_)
            throw std::runtime_error("Cannot map PCI memory space");
    }

    // reset hardware state
    void reset() { *cmd2addr(TM_REQ_RESET, tnum_) = 0; }

  public:
    const bool is_lock_free() const {return true;}

    fifo(boost::uint64_t sz, boost::uint64_t tnum)
      : tnum_(tnum)
    {
        map_memory();
        reset();
        boost::uint64_t qs = *cmd2addr(TM_REQ_GETSIZE);
        if (qs < sz)
            throw std::runtime_error("Hardware queue does not have sufficent capacity");
    }

    ~fifo()
    {
        if (dev_)
        {
            delete dev_;
            pci_system_cleanup();
        }
    }

    bool empty()
    {
        return 0ULL == *cmd2addr(TM_REQ_GETCNT, tnum_);
    }

    bool push(T const& t)
    {
        *cmd2addr(TM_REQ_SETLAST, tnum_) = reinterpret_cast<boost::uint64_t>(t);
        return true;
    }

    bool pop(T& t)
    {
        t = *reinterpret_cast<T>(*cmd2addr(TM_REQ_GETHEAD, tnum_));
        return true;
    }
};

}}

namespace threads { namespace policy
{

struct hardware_fifo;

template <typename T, typename Queuing>
struct basic_hardware_queue_backend
{
    typedef Queuing container_type;
    typedef T value_type;
    typedef T& reference;
    typedef T const& const_reference;
    typedef boost::uint64_t size_type;

    basic_lockfree_queue_backend(
        size_type initial_size = 0
      , size_type num_thread = size_type(-1)
        )
      : queue_(initial_size, num_thread)
    {}

    bool push(const_reference val)
    {
        return queue_.push(val);
    }

    bool pop(reference val, bool steal = true)
    {
        return queue_.pop(val);
    }

    bool empty()
    {
        return queue_.empty();
    }

  private:
    container_type queue_;
};

struct hardware_fifo
{
    template <typename T>
    struct apply
    {
        typedef basic_hardware_queue_backend<
            T, hpx::util::hardware::fifo<T>
        > type;
    };
};

}}}

#endif // HPX_3ED714E9_F359_423C_93DC_B819A313BBAF

