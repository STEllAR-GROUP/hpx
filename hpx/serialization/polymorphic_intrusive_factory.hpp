//  Copyright (c) 2014 Anton Bikineev
//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_POLYMORPHIC_INTRUSIVE_FACTORY_HPP
#define HPX_SERIALIZATION_POLYMORPHIC_INTRUSIVE_FACTORY_HPP

#include <hpx/config.hpp>
#include <hpx/util/jenkins_hash.hpp>
#include <hpx/util/static.hpp>

#include <boost/noncopyable.hpp>
#include <boost/unordered_map.hpp>
#include <boost/mpl/bool.hpp>

namespace hpx { namespace serialization {

  struct unique_data_type
  {
    boost::uint64_t hash;
    std::string name;
  };

  class HPX_EXPORT polymorphic_intrusive_factory: boost::noncopyable
  {
  public:
    typedef hpx::util::jenkins_hash::size_type key_type;
    typedef void* (*ctor_type) ();
    typedef boost::unordered_map<key_type,
            std::pair<std::string, ctor_type> > ctor_map_type;

    static polymorphic_intrusive_factory& instance()
    {
      hpx::util::static_<polymorphic_intrusive_factory> factory;
      return factory.get();
    }

    void register_class(const unique_data_type& data, ctor_type fun)
    {
      map_[data.hash] = std::make_pair(data.name, fun);
    }

    template <class T>
    T* create(const unique_data_type& data) const
    {
      return static_cast<T*>(locate(data)->second.second());
    }

  private:
    polymorphic_intrusive_factory()
    {
    }

    typename ctor_map_type::const_iterator locate( // TODO
        const unique_data_type& unique_data) const
    {
      typedef std::pair<
        typename ctor_map_type::const_iterator,
        typename ctor_map_type::const_iterator> equal_range_type;

      equal_range_type range = map_.equal_range(unique_data.hash);
      if (range.first != range.second)
      {
        typename ctor_map_type::const_iterator it = range.first;
        if (++it == range.second)
        {
          // there is only one math in the map
          return range.first;
        }

        //there is more than one entry with the same hash in the map
        for (it = range.first; it != range.second; ++it)
        {
          if ((*it).second.first == unique_data.name)
            return it;
        }
      }
      return map_.end();
    }

    friend struct hpx::util::static_<polymorphic_intrusive_factory>;

    ctor_map_type map_;
  };

}}

#define HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS_WITH_NAME(Class, Name)        \
  virtual ::hpx::serialization::unique_data_type                              \
  hpx_serialization_get_unique_data() const                                   \
  {                                                                           \
    static boost::uint64_t const hash =                                       \
      hpx::util::jenkins_hash()(Name);                                        \
                                                                              \
    return ::hpx::serialization::unique_data_type{ hash, Name };              \
  }                                                                           \
  static void* factory_function()                                             \
  {                                                                           \
    return new Class;                                                         \
  }                                                                           \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME(Class, Name)                  \
  virtual void load(hpx::serialization::input_archive& ar, unsigned n)        \
  {                                                                           \
    serialize<hpx::serialization::input_archive>(ar, n);                      \
  }                                                                           \
  virtual void save(hpx::serialization::output_archive& ar, unsigned n) const \
  {                                                                           \
    static bool register_class = (                                            \
      hpx::serialization::polymorphic_intrusive_factory::instance().          \
        register_class(                                                       \
          Class::hpx_serialization_get_unique_data(),                         \
          &Class::factory_function                                            \
        ),                                                                    \
      true                                                                    \
    );                                                                        \
    const_cast<Class*>(this)->                                                \
      serialize<hpx::serialization::output_archive>(ar, n);                   \
  }                                                                           \
  HPX_SERIALIZATION_SPLIT_MEMBER();                                           \
  HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS_WITH_NAME(Class, Name);             \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_SPLITTED(Class, Name)         \
  virtual void load(hpx::serialization::input_archive& ar, unsigned n)        \
  {                                                                           \
    load<hpx::serialization::input_archive>(ar, n);                           \
  }                                                                           \
  virtual void save(hpx::serialization::output_archive& ar, unsigned n) const \
  {                                                                           \
    static bool register_class = (                                            \
      hpx::serialization::polymorphic_intrusive_factory::instance().          \
        register_class(                                                       \
          Class::hpx_serialization_get_unique_data(),                         \
          &Class::factory_function                                            \
        ),                                                                    \
      true                                                                    \
    );                                                                        \
    save<hpx::serialization::output_archive>(ar, n);                          \
  }                                                                           \
  HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS_WITH_NAME(Class, Name);             \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC_ABSTRACT(Class)                         \
  virtual void load(hpx::serialization::input_archive& ar, unsigned n)        \
  {                                                                           \
    serialize<hpx::serialization::input_archive>(ar, n);                      \
  }                                                                           \
  virtual void save(hpx::serialization::output_archive& ar, unsigned n) const \
  {                                                                           \
    const_cast<Class*>(this)->                                                \
      serialize<hpx::serialization::output_archive>(ar, n);                   \
  }                                                                           \
  HPX_SERIALIZATION_SPLIT_MEMBER()                                            \
  virtual ::hpx::serialization::unique_data_type                              \
    hpx_serialization_get_unique_data() const = 0;                            \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC_ABSTRACT_SPLITTED(Class)                \
  virtual void load(hpx::serialization::input_archive& ar, unsigned n)        \
  {                                                                           \
    load<hpx::serialization::input_archive>(ar, n);                           \
  }                                                                           \
  virtual void save(hpx::serialization::output_archive& ar, unsigned n) const \
  {                                                                           \
    save<hpx::serialization::output_archive>(ar, n);                          \
  }                                                                           \
  virtual ::hpx::serialization::unique_data_type                              \
    hpx_serialization_get_unique_data() const = 0;                            \
/**/

#define HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS(Class)                        \
  HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS_WITH_NAME(                          \
      Class, BOOST_STRINGIZE(Class))                                          \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC(Class)                                  \
  HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME(Class, BOOST_STRINGIZE(Class))      \

#define HPX_SERIALIZATION_POLYMORPHIC_SPLITTED(Class)                         \
  HPX_SERIALIZATION_POLYMORPHIC_WITH_NAME_SPLITTED(                           \
      Class, BOOST_STRINGIZE(Class))                                          \
/**/

#endif
