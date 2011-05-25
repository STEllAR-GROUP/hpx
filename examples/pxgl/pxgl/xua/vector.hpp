// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_VECTORS_BLOCK_VECTOR_20100915T1735)
#define PXGL_VECTORS_BLOCK_VECTOR_20100915T1735

#include <hpx/hpx.hpp>

#include "../../pxgl/pxgl.hpp"
#include "../../pxgl/util/scoped_use.hpp"
#include "../../pxgl/util/component.hpp"

// Define logging helper
#define LVECT_LOG_fatal 1
#define LVECT_LOG_info 0
#define LVECT_LOG__ping 0

#if LVECT_LOG_ping == 1
#  define LVECT_ping(major,minor) YAP_now(major,minor)
#else
#  define LVECT_ping(major,minor) do {} while(0)
#endif

#if LVECT_LOG_info == 1
#define LVECT_info(str,...) YAPs(str,__VA_ARGS__)
#else
#define LVECT_info(str,...) do {} while(0)
#endif

#if LVECT_LOG_fatal == 1
#define LVECT_fatal(str,...) YAPs(str,__VA_ARGS__)
#else
#define LVECT_fatal(str,...) do {} while(0)
#endif

////////////////////////////////////////////////////////////////////////////////
// Prototypes
namespace pxgl { namespace xua {
  template <typename Distribution, typename Item>
  class vector;
}}

////////////////////////////////////////////////////////////////////////////////
// Server interface
namespace pxgl { namespace xua { namespace server {
  template <typename Distribution, typename Item>
  class vector
    : public HPX_MANAGED_BASE_2(vector, Distribution, Item)
  {
  public:
    enum actions
    {
      // Construction
      vector_construct,
      vector_replicate,
      // Initialization
      vector_constructed,
      vector_init,
      vector_init_sync,
      vector_init_items,
      // Use
      vector_ready,
      vector_ready_all,
      vector_size,
      vector_get_distribution,
      vector_items,
      vector_clear,
      vector_clear_member,
      vector_local_to,
    };

    ////////////////////////////////////////////////////////////////////////////
    // Common types
    typedef hpx::naming::id_type id_type;
    typedef std::vector<id_type> ids_type;

    typedef hpx::naming::gid_type gid_type;

    typedef unsigned long size_type;

    ////////////////////////////////////////////////////////////////////////////
    // Associated types
    typedef Distribution distribution_type;

    typedef Item item_type;
    typedef std::vector<item_type> items_type;

    typedef pxgl::xua::vector<Distribution, Item> vector_client_type;

    ////////////////////////////////////////////////////////////////////////////
    // Construction interface
    vector()
      : items_(0),
        size_(0),
        d_ary_(4),
        outstanding_syncs_(0),
        me_index_(0),
        here_(hpx::get_runtime().get_process().here()),
        constructed_(false),
        initialized_(false)
    {
      use_feb_.set(feb_data_type(1));
    }

    ~vector()
    {
      LVECT_ping("Vector", "Deconstructing");
    }

    //!
    //! \brief Build out the distributed vector.
    //!
    //! \param me [in] The GID of this vector.
    //! \param distribution [in] The distribution to use for this vector.
    //!
    void construct(
        id_type const & me, 
        distribution_type const & distribution)
    {
      {
        pxgl::util::scoped_use l(use_feb_);

        assert(!initialized_);

        me_ = me;
        distribution_ = distribution;

        typedef typename distribution_type::locality_ids_type locality_ids_type;
        locality_ids_type const & locales = distribution_.coverage();
        size_type const extent = locales.size();

        siblings_ = std::vector<vector_client_type>(extent);
        for (size_type i = 0; i < extent; i++)
        {
          if (locales[i] != here_)
          {
            siblings_[i].create(locales[i]);
          }
          else
          {
            siblings_[i] = vector_client_type(me_);
            me_index_ = i;
          }
        }

        // Collect ids for siblings
        // Note: we include this actor in the collection of siblings
        ids_type sibling_ids(distribution.size());
        for (size_type i =0; i < extent; i++)
        {
          sibling_ids[i] = siblings_[i].get_gid();
        }

        // Construct siblings
        for (size_type i =0; i < extent; i++)
        {
          if (locales[i] != here_)
          {
            siblings_[i].replicate(distribution, sibling_ids);
          }
        }
        
        constructed_feb_.set(feb_data_type(1));
        constructed_ = true;
      }
    }

    typedef hpx::actions::action2<
        vector, 
        vector_construct, 
            id_type const &, 
            distribution_type const &, 
        &vector::construct
    > construct_action;

    //!
    //! \brief Build out a distributed (sub-)vector.
    //!
    //! \param distribution [in] The distribution to use for this vector.
    //! \param sibling_ids [in] A collection of GIDs for each sibling vector.
    //!
    void replicate(
        distribution_type const & distribution, 
        ids_type const & sibling_ids)
    {
      {
        pxgl::util::scoped_use l(use_feb_);

        assert(!initialized_);

        // Set distribution
        distribution_ = distribution;

        typedef typename distribution_type::locality_ids_type locality_ids_type;
        locality_ids_type const & locales = distribution_.coverage();
        size_type const extent = locales.size();

        // Set siblings and me
        siblings_ = std::vector<vector_client_type>(extent);
        for (size_type i = 0; i < extent; i++)
        {
          siblings_[i] = vector_client_type(sibling_ids[i]);

          if (locales[i] == here_)
          {
            me_ = sibling_ids[i];
            me_index_ = i;
          }
        }

        // Set as constructed
        constructed_feb_.set(feb_data_type(1));
        constructed_ = true;
      }
    }
    
    typedef hpx::actions::action2<
        vector, 
        vector_replicate, 
            distribution_type const &, 
            ids_type const &,
        &vector::replicate
    > replicate_action;

    //!
    //! \brief Synchronizes on construction of the vector.
    //!
    void constructed()
    {
      while (!constructed_)
      {
        feb_data_type d;
        constructed_feb_.read(d);

        if (1 == d.which())
        {
          error_type e = boost::get<error_type>(d);
          boost::rethrow_exception(e);
        }
      }
    }

    typedef hpx::actions::action0<
        vector, 
        vector_constructed, 
        &vector::constructed
    > constructed_action;

    ////////////////////////////////////////////////////////////////////////////
    // Initialization interface

    //!
    //! \brief Initialize this vector
    //! \param items [in] A collection of items to add to this (sub-)vector.
    //!
    inline size_type find_num_followers(void) const
    {
      size_type const last_leader = 
          (size_type)(((siblings_.size()-1) - 1) / d_ary_);

      if (me_index_ < last_leader)
        return d_ary_;
      else if (me_index_ == last_leader)
        return (siblings_.size() - 1) - (last_leader * d_ary_);
      else
        return 0;
    }

    inline size_type find_my_leader(void) const
    {
      return (size_type)((me_index_ - 1) / d_ary_);
    }

    void init(items_type const & items)
    {
      // Wait for member to be constructed
      constructed();

      {
        pxgl::util::scoped_use l(use_feb_);

        // Initialize only if items were not already set with init_items()
        if (items_.size() == 0)
        {
          items_ = items;
        }
        else
        {
          LVECT_fatal("Using %lu items I have.\n", items_.size());
        }

        size_ += items_.size();

        outstanding_syncs_ += 1;

        size_type const num_followers = find_num_followers();
        if (num_followers + 1 == outstanding_syncs_)
        {
          if (me_index_ != 0)
          {
            size_type const my_leader = find_my_leader();
            size_ = siblings_[my_leader].init_sync(size_);
          }

          initialized_feb_.set(feb_data_type(1));
          initialized_ = true;
        }

        // Special case of only a single locality
        if (siblings_.size() == 1)
        {
          initialized_feb_.set(feb_data_type(1));
          initialized_ = true;
        }
      }
    }

    typedef hpx::actions::action1<
        vector, 
        vector_init, 
          items_type const &, 
        &vector::init
    > init_action;

    //!
    //! \brief Synchronize initialization of the distributed vector.
    //!
    //! \param local_size [in] The size of the callers local segment of
    //!        the vector.
    //!
    size_type init_sync(size_type local_size)
    {
      {
        pxgl::util::scoped_use l(use_feb_);

        size_ += local_size;
        outstanding_syncs_ += 1;

        size_type const num_followers = find_num_followers();
        if (num_followers + 1 == outstanding_syncs_)
        {
          if (me_index_ != 0)
          {
            size_type const my_leader = find_my_leader();
            size_ = siblings_[my_leader].init_sync(size_);
          }

          initialized_feb_.set(feb_data_type(1));
          initialized_ = true;
        }
      }

      return size_;
    }

    typedef hpx::actions::result_action1<
        vector, 
            size_type, 
        vector_init_sync, 
            size_type,
        &vector::init_sync
    > init_sync_action;

    //!
    //! \brief Synchronize on the initialization of the vector.
    //!
    void ready(void)
    {
      while (!initialized_)
      {
        feb_data_type d;
        initialized_feb_.read(d);

        if (1 == d.which())
        {
          error_type e = boost::get<error_type>(d);
          boost::rethrow_exception(e);
        }
      }
    }

    typedef hpx::actions::action0<
        vector, vector_ready, 
        &vector::ready
    > ready_action;

    void ready_all(void)
    {
      constructed();

      BOOST_FOREACH(vector_client_type sibling, siblings_)
      {
        sibling.ready();
      }
    }

    typedef hpx::actions::action0<
        vector, 
        vector_ready_all, 
        &vector::ready_all
    > ready_all_action;

    ////////////////////////////////////////////////////////////////////////////
    // Usage interface

    //!
    //! \brief Returns the size of the distributed vector.
    //!
    size_type size(void)
    {
      ready();

      return size_;
    }

    typedef hpx::actions::result_action0<
        vector, 
            size_type, 
        vector_size, 
        &vector::size
    > size_action;

    //!
    //! \brief Returns the distribution of the distributed vector.
    //!
    distribution_type get_distribution(void)
    {
      constructed();

      return distribution_;
    }

    typedef hpx::actions::result_action0<
        vector, distribution_type, vector_get_distribution, 
        &vector::get_distribution
    > get_distribution_action;

    //!
    //! \brief Returns the collection of items held by this vector.
    //!
    items_type * items(void)
    {
      ready();

      return &items_;
    }

    typedef hpx::actions::result_action0<
        vector, 
            items_type *, 
        vector_items,
        &vector::items
    > items_action;

    items_type * init_items(size_type size)
    {
      //not_ready();
      assert(!initialized_);

      constructed();

      {
        pxgl::util::scoped_use l(use_feb_);

        if (items_.size() == 0)
        {
          items_.resize(size);
        }
      }

      return &items_;
    }

    typedef hpx::actions::result_action1<
        vector, 
            items_type *, 
        vector_init_items,
            size_type,
        &vector::init_items
    > init_items_action;

    //!
    //! \briefs Clear the contents of the vector
    //!
    //! Clear the contents of the distributed vector and free all associated memory.
    //!
    //! This frees up all the internal references to the member components.
    //!
    void clear(void)
    {
      ready();

      scoped_lock l(this);

      LVECT_ping("Vector", "Clearing");

      // Tell all of the other members to clear
      BOOST_FOREACH(vector_client_type sibling, siblings_)
      {
        if (sibling.get_gid() != me_)
        {
          sibling.clear_member();
        }
      }

      siblings_.clear();
      me_ = hpx::naming::invalid_id;
    }

    typedef hpx::actions::action0<
        vector, vector_clear,
        &vector::clear
    > clear_action;

    void clear_member(void)
    {
      ready();

      scoped_lock l(this);

      LVECT_ping("Vector", "Clearing member");

      siblings_.clear();
      me_ = hpx::naming::invalid_id;
    }

    typedef hpx::actions::action0<
        vector, vector_clear_member,
        &vector::clear_member
    > clear_member_action;

    //!
    //! \brief Return the GID of the sub-vector on a certain locality.
    //!
    //! \param index the logical ID of a locality in this coverage.
    //!
    id_type local_to(size_type index)
    {
      constructed();

      return siblings_[index].get_gid();
    }

    typedef hpx::actions::result_action1<
        vector, id_type, vector_local_to,
        size_type,
        &vector::local_to
    > local_to_action;

  private:
    void set_branching_factor(void)
    {
      d_ary_ = boost::lexical_cast<size_type>(
          hpx::get_runtime().get_config().get_entry("vector.d_ary", d_ary_));
      assert(d_ary_ != 0);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Data members
    items_type items_;
    size_type size_;

    // Branching factor of d-ary initialization tree
    size_type d_ary_;

    size_type outstanding_syncs_;

    id_type me_;
    size_type me_index_;
    id_type here_;
    distribution_type distribution_;
    std::vector<vector_client_type> siblings_;

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

    // Use to block threads around critical sections
    hpx::util::full_empty<feb_data_type> use_feb_;
  };
}}}

////////////////////////////////////////////////////////////////////////////////
// Stubs interface
namespace pxgl { namespace xua { namespace stubs {
  template <typename Distribution, typename Item>
  struct vector
    : HPX_STUBS_BASE_2(vector, Distribution, Item)
  {
    ////////////////////////////////////////////////////////////////////////////
    // Associated types
    typedef server::vector<Distribution,Item> server_type;
    typedef typename server_type::id_type id_type;
    typedef typename server_type::ids_type ids_type;
    typedef typename server_type::size_type size_type;
    typedef typename server_type::item_type item_type;
    typedef typename server_type::items_type items_type;
    typedef typename server_type::distribution_type distribution_type;

    ////////////////////////////////////////////////////////////////////////////
    // Construction interface
    
    //!
    //! Asynchronous action.
    //!
    static void construct(
        id_type const & id, 
        id_type const & me, 
        Distribution const & distribution)
    {
      typedef typename server_type::construct_action action_type;
      hpx::applier::apply<action_type>(id, me, distribution);
    }

    static void replicate(
        id_type const & id,
        distribution_type const & distribution,
        ids_type const & sibling_ids)
    {
      typedef typename server_type::replicate_action action_type;
      hpx::applier::apply<action_type>(id, distribution, sibling_ids);
    }

    static void sync_constructed(id_type id)
    {
      typedef typename server_type::constructed_action action_type;
      hpx::lcos::eager_future<action_type>(id).get();
    }

    ////////////////////////////////////////////////////////////////////////////
    // Initialization interface
    static void init(id_type const & id)
    {
      typedef typename server_type::init_action action_type;

      hpx::applier::apply<action_type>(id, items_type());
    }

    static void init(
        id_type const & id, 
        items_type const & items)
    {
      typedef typename server_type::init_action action_type;
      hpx::applier::apply<action_type>(id, items);
    }

    static size_type init_sync(
        id_type const & id, 
        size_type local_size)
    {
      typedef typename server_type::init_sync_action action_type;
      return hpx::lcos::eager_future<action_type>(id, local_size).get();
    }

    static void ready(id_type const & id)
    {
      typedef typename server_type::ready_action action_type;
      hpx::lcos::eager_future<action_type>(id).get();
    }

    static void ready_all(id_type const & id)
    {
      typedef typename server_type::ready_all_action action_type;
      hpx::lcos::eager_future<action_type>(id).get();
    }

    ////////////////////////////////////////////////////////////////////////////
    // Usage interface
    static size_type size(id_type const & id)
    {
      typedef typename server_type::size_action action_type;
      return hpx::lcos::eager_future<action_type>(id).get();
    }

    static Distribution get_distribution(id_type const & id)
    {
      typedef typename server_type::get_distribution_action action_type;
      return hpx::lcos::eager_future<action_type>(id).get();
    }

    static items_type * items(id_type const & id)
    {
      typedef typename server_type::items_action action_type;
      return hpx::lcos::eager_future<action_type>(id).get();
    }

    static items_type * sync_init_items(
        id_type const & id,
        size_type const size)
    {
      typedef typename server_type::init_items_action action_type;
      return hpx::lcos::eager_future<action_type>(id, size).get();
    }

    static void clear(id_type id)
    {
      typedef typename server_type::clear_action action_type;
      hpx::lcos::eager_future<action_type>(id).get();
    }

    static void clear_member(id_type id)
    {
      typedef typename server_type::clear_member_action action_type;
      hpx::lcos::eager_future<action_type>(id).get();
    }

    static id_type local_to(
        id_type const & id, 
        size_type index)
    {
      typedef typename server_type::local_to_action action_type;
      return hpx::lcos::eager_future<action_type>(id, index).get();
    }
  };
}}}

////////////////////////////////////////////////////////////////////////////////
// Client interface
namespace pxgl { namespace xua {
  template <typename Distribution, typename Item>
  class vector
    : public HPX_CLIENT_BASE_2(vector, Distribution, Item)
  {
  private:
    typedef HPX_CLIENT_BASE_2(vector, Distribution, Item) base_type;

  public:
    ////////////////////////////////////////////////////////////////////////////
    // Associated types
    typedef typename stubs::vector<Distribution,Item> stubs_type;
    typedef typename stubs_type::id_type id_type;
    typedef typename stubs_type::ids_type ids_type;
    typedef typename stubs_type::item_type item_type;
    typedef typename stubs_type::items_type items_type;
    typedef typename stubs_type::size_type size_type;
    typedef typename stubs_type::distribution_type distribution_type;

    ////////////////////////////////////////////////////////////////////////////
    // Construction
    vector()
    {}

    vector(id_type id)
      : base_type(id)
    {}

    //!
    //! Initiates construction of a distributed vector.
    //!
    //! Asynchronous action.
    //!
    void construct(
        id_type const & me, 
        Distribution const & distribution) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::construct(this->gid_, me, distribution);
    }

    void replicate(
        Distribution const & distribution, 
        ids_type const & sibling_ids) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::replicate(this->gid_, distribution, sibling_ids);
    }

    void sync_constructed(void)
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::sync_constructed(this->gid_);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Initialization
    void init(void) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::init(this->gid_);
    }

    void init(items_type const & items) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::init(this->gid_, items);
    }

    size_type init_sync(size_type local_size) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::init_sync(this->gid_, local_size);
    }

    void ready(void) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::ready(this->gid_);
    }

    void ready_all(void) const
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::ready_all(this->gid_);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Usage interface
    size_type size(void) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::size(this->gid_);
    }

    Distribution get_distribution(void) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::get_distribution(this->gid_);
    }

    items_type * items(void) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::items(this->gid_);
    }

    items_type * sync_init_items(size_type const size) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::sync_init_items(this->gid_, size);
    }

    void clear(void)
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::clear(this->gid_);
    }

    void clear_member(void)
    {
      BOOST_ASSERT(this->gid_);
      this->base_type::clear_member(this->gid_);
    }

    id_type local_to(size_type index) const
    {
      BOOST_ASSERT(this->gid_);
      return this->base_type::local_to(this->gid_, index);
    }
  };
}}

#endif

