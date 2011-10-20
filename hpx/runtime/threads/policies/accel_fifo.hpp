// Copyright (c) 2010 Maciej Brodowicz
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "pci.hh"
#include "accel_defs.h"


namespace accel
{
  // wrapper for hardware fifo;
  // operates on PODs up to 8 bytes in size
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

  protected:
    // hw request to address mapping;
    // requires OS thread number (small integer) to select hardware
    // state necessary to manage split PCI requests
    inline uint64_t *cmd2addr(unsigned cmd, size_t tnum)
    {
      return reinterpret_cast<uint64_t *>(base_+((cmd | (tnum<<TM_REQ_BITS))<<3));
    }

    // find the device and map the first available memory BAR
    void map_memory()
    {
      if (pci_system_init() != 0)
    throw std::runtime_error(std::string("PCI system initialization failed"));
      // match any Xilinx board
      // (temporary, as there are also boards from Avnet and HTG)
      dev_ = new pci::Device(XilinxID, PCI_MATCH_ANY);
      if (!dev_ || !dev_->valid())
    throw std::runtime_error(std::string("Failed to find the PCI FPGA board"));
      base_ = reinterpret_cast<char *>(dev_->region(PCI_BAR_ANY));
      if (!base_)
    throw std::runtime_error(std::string("Cannot map PCI memory space"));
      printf("using accelerated queue at %p\n", base_);
    }

    // reset hardware state
    inline void reset(size_t tnum = 0) {*cmd2addr(TM_REQ_RESET, tnum) = 0;}

  public:
    const bool is_lock_free() const {return true;}

    fifo()
    {
      map_memory();
      reset();
    }

    explicit fifo(std::size_t sz)
    {
      map_memory();
      reset();
      uint64_t qs = *cmd2addr(TM_REQ_GETSIZE);
      if (qs < sz)
    throw std::runtime_error(std::string("Hardware queue does not have sufficent capacity"));
    }

    ~fifo()
    {
      if (dev_)
      {
    delete dev_;
    pci_system_cleanup();
      }
    }

    bool empty(size_t tnum)// = 0)
    {
      return 0ULL == *cmd2addr(TM_REQ_GETCNT, tnum);
    }

    bool enqueue(T const& t, size_t tnum)// = 0)
    {
      //printf("enq%ld\n", tnum);
      *cmd2addr(TM_REQ_SETLAST, tnum) = reinterpret_cast<uint64_t>(t);
      return true;
    }
    bool dequeue(T *t, size_t tnum)// = 0)
    {
      //printf("deq%ld\n", tnum);
      *t = reinterpret_cast<T>(*cmd2addr(TM_REQ_GETHEAD, tnum));
      return *t != 0;
    }
  };
}
