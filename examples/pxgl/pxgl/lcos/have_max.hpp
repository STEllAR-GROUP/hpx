// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_LOCS_HAVE_MAX_20110120T1436)
#define PXGL_LOCS_HAVE_MAX_20110120T1436

#include <pxgl/pxgl.hpp>
#include <pxgl/util/component.hpp>
#include <pxgl/util/scoped_use.hpp>

#define LLHM_info(str) 
#define LLHM_debug(str) 
#define LLHM_fatal(str) 

////////////////////////////////////////////////////////////////////////////////
// Prototypes
namespace pxgl { namespace lcos {
  template <typename Item>
  class have_max;
}}

////////////////////////////////////////////////////////////////////////////////
// Server interface
namespace pxgl { namespace lcos { namespace server {
  template <typename Item>
  class have_max
    : public HPX_MANAGED_BASE_1(have_max, Item)
  {
  public:
    enum actions
    {
      // Construction
      have_max_construct,
      // Initialization
      // Use
      have_max_ready,
      have_max_signal,
    };

    ////////////////////////////////////////////////////////////////////////////
    // Common types
    typedef hpx::naming::id_type id_type;
    typedef std::vector<id_type> ids_type;

    typedef unsigned long size_type;

    ////////////////////////////////////////////////////////////////////////////
    // Associated types
    typedef Item item_type;
    typedef std::vector<item_type> items_type;

    typedef pxgl::lcos::have_max<Item> have_max_client_type;

    ////////////////////////////////////////////////////////////////////////////
    // Construction interface
    have_max()
      : num_items_(0),
        items_(0),
        max_item_(std::numeric_limits<Item>::min()),
        constructed_(false),
        initialized_(false)
    {
      use_feb_.set(feb_data_type(1));
    }

    //!
    //! \brief Build out the distributed have_max.
    //!
    //! \param me [in] The GID of this have_max.
    //! \param distribution [in] The distribution to use for this have_max.
    //!
    void construct(size_type num_items)
    {
      {
        pxgl::util::scoped_use l(use_feb_);

        assert(!initialized_);

        num_items_ = num_items;

        LLHM_info(num_items_ << " participants in have-max LCO")

        constructed_feb_.set(feb_data_type(1));
        constructed_ = true;
        LLHM_debug("Setting variable have_max as constructed")

        // For the LCO, construction is initialization
        initialized_feb_.set(feb_data_type(1));
        initialized_ = true;
        LLHM_debug("Setting variable have_max as initialized")
      }
    }

    typedef hpx::actions::action1<
        have_max, 
        have_max_construct, 
            size_type,
        &have_max::construct
    > construct_action;

    //!
    //! \brief Synchronizes on construction of the have_max.
    //!
    void constructed()
    {
      while (!constructed_)
      {
        LLHM_info("Waiting on variable have_max to be constructed.")

        feb_data_type d;
        constructed_feb_.read(d);

        if (1 == d.which())
        {
          error_type e = boost::get<error_type>(d);
          boost::rethrow_exception(e);
        }

        LLHM_info("Got variable have_max as constructed")
      }
    }

    ////////////////////////////////////////////////////////////////////////////
    // Initialization interface

    //!
    //! \brief Synchronize on the initialization of the have_max.
    //!
    void ready()
    {
      while (!initialized_)
      {
        LLHM_info("Waiting on variable have_max to be initialized.")

        feb_data_type d;
        initialized_feb_.read(d);

        if (1 == d.which())
        {
          error_type e = boost::get<error_type>(d);
          boost::rethrow_exception(e);
        }

        LLHM_info("Got variable have_max as initialized")
      }
    }

    typedef hpx::actions::action0<
        have_max, have_max_ready, 
        &have_max::ready
    > ready_action;

    ////////////////////////////////////////////////////////////////////////////
    // Usage interface

    bool signal(item_type value)
    {
      ready();

      bool outstanding_values;
      {
        pxgl::util::scoped_use l(use_feb_);

        // Decrement count of outstanding items
        num_items_--;
        outstanding_values = (0 < num_items_);

        // Update max value
        if (value > max_item_)
        {
          max_item_ = value;
        }
      }

      if (outstanding_values)
      {
        if (value < max_item_)
        {
          return false;
        }
        else
        {
          // Suspend this thread
          hpx::threads::thread_self& self = hpx::threads::get_self();
          hpx::threads::thread_id_type id = self.get_thread_id();
          wait_queue_.enqueue(id);

          self.yield(hpx::threads::suspended);
        }
      }
      else
      {
        // Activate suspending threads
        hpx::threads::thread_id_type id = 0;
        while (wait_queue_.dequeue(&id))
            hpx::threads::set_thread_state(id, hpx::threads::pending);
      }

      return value == max_item_;
    }

    typedef hpx::actions::result_action1<
        have_max, 
            bool, 
        have_max_signal,
            item_type,
        &have_max::signal
    > signal_action;

  private:
    ////////////////////////////////////////////////////////////////////////////
    // Data members
    size_type num_items_;
    items_type items_;
    item_type max_item_;

    ////////////////////////////////////////////////////////////////////////////
    // Synchronization members
    struct tag {};
    typedef hpx::util::spinlock_pool<tag> mutex_type;
    typedef typename mutex_type::scoped_lock scoped_lock;

    typedef int result_type;
    typedef boost::exception_ptr error_type;
    typedef boost::variant<result_type, error_type> feb_data_type;

    // Used to suspend calling threads until data structure is constructed
    // Note: this is required because we cannot pass arguments to the
    // component constructor
    bool constructed_;
    hpx::util::full_empty<feb_data_type> constructed_feb_;

    // Used to suspend calling threads until data structure is initialized
    bool initialized_;
    hpx::util::full_empty<feb_data_type> initialized_feb_;

    // Used to suspend signalling threads
    boost::lockfree::fifo<hpx::threads::thread_id_type> wait_queue_;

    // Use to block threads around critical sections
    hpx::util::full_empty<feb_data_type> use_feb_;
  };
}}}

////////////////////////////////////////////////////////////////////////////////
// Stubs interface
namespace pxgl { namespace lcos { namespace stubs {
  template <typename Item>
  struct have_max
    : HPX_STUBS_BASE_1(have_max, Item)
  {
    ////////////////////////////////////////////////////////////////////////////
    // Associated types
    typedef server::have_max<Item> server_type;
    typedef typename server_type::id_type id_type;
    typedef typename server_type::size_type size_type;
    typedef typename server_type::item_type item_type;
    typedef typename server_type::items_type items_type;

    ////////////////////////////////////////////////////////////////////////////
    // Construction interface
    static void construct(
        id_type const & id, 
        size_type num_items)
    {
      typedef typename server_type::construct_action action_type;
      hpx::applier::apply<action_type>(id, num_items);
    }
    
    ////////////////////////////////////////////////////////////////////////////
    // Initialization interface

    ////////////////////////////////////////////////////////////////////////////
    // Usage interface
    static bool signal(
        id_type const & id, 
        item_type value)
    {
      typedef typename server_type::signal_action action_type;
      return hpx::lcos::eager_future<action_type>(id, value).get();
    }
  };
}}}

////////////////////////////////////////////////////////////////////////////////
// Client interface
namespace pxgl { namespace lcos {
  template <typename Item>
  class have_max
    : public HPX_CLIENT_BASE_1(have_max, Item)
  {
  private:
    typedef HPX_CLIENT_BASE_1(have_max, Item) base_type;

  public:
    ////////////////////////////////////////////////////////////////////////////
    // Associated types
    typedef Item item_type;

    typedef typename stubs::have_max<Item> stubs_type;
    typedef typename stubs_type::id_type id_type;
    typedef typename stubs_type::items_type items_type;
    typedef typename stubs_type::size_type size_type;

    ////////////////////////////////////////////////////////////////////////////
    // Construction
    have_max()
    {}

    have_max(id_type id)
      : base_type(id)
    {}

    void construct(size_type num_items) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::construct(this->gid_, num_items);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Initialization

    ////////////////////////////////////////////////////////////////////////////
    // Usage interface
    bool signal(item_type value) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::signal(this->gid_, value);
    }
  };
}}

#endif

