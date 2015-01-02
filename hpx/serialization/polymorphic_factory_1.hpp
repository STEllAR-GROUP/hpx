//  Copyright (c) 2014 Anton Bikineev
//
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_POLYMORPHIC_FACTORY_1_HPP
#define HPX_SERIALIZATION_POLYMORPHIC_FACTORY_1_HPP

#include <hpx/config.hpp>
#include <hpx/util/jenkins_hash.hpp>
#include <hpx/util/static.hpp>

#include <boost/noncopyable.hpp>
#include <boost/unordered_map.hpp>
#include <boost/mpl/bool.hpp>

namespace hpx { namespace serialization {

  class HPX_EXPORT polymorphic_factory: boost::noncopyable
  {
  public:
    typedef hpx::util::jenkins_hash::size_type size_type;
    typedef void* (*ctor_type) ();
    typedef boost::unordered_map<size_type, ctor_type> ctor_map;

    static polymorphic_factory& instance()
    {
      hpx::util::static_<polymorphic_factory> factory;
      return factory.get();
    }

    void register_class(boost::uint64_t hash, ctor_type fun)
    {
      map[hash] = fun;
    }

    void* create(size_type hash)
    {
      return map.at(hash)();
    }

  private:
    polymorphic_factory()
    {
    }

    friend hpx::util::static_<polymorphic_factory>;

    ctor_map map;
  };

}}

#define HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS(Class)                        \
  virtual boost::uint64_t hpx_serialization_get_hash() const                  \
  {                                                                           \
    /* assumes the class name is the unique name in the file  */              \
    /* addition of __LINE__ can also be considered            */              \
    static boost::uint64_t const hash =                                       \
      hpx::util::jenkins_hash()(                                              \
        BOOST_PP_STRINGIZE(Class)                                             \
        __FILE__                                                              \
      );                                                                      \
    return hash;                                                              \
  }                                                                           \
  static void* factory_function()                                             \
  {                                                                           \
    return new Class;                                                         \
  }                                                                           \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC(Class)                                  \
  virtual void load(hpx::serialization::input_archive& ar, unsigned n)        \
  {                                                                           \
    serialize<hpx::serialization::input_archive>(ar, n);                      \
  }                                                                           \
  virtual void save(hpx::serialization::output_archive& ar, unsigned n) const \
  {                                                                           \
    static bool register_class = (                                            \
      hpx::serialization::polymorphic_factory::instance().                    \
        register_class(                                                       \
          Class::hpx_serialization_get_hash(),                                \
          &Class::factory_function                                            \
        ),                                                                    \
      true                                                                    \
    );                                                                        \
    const_cast<Class*>(this)->                                                \
      serialize<hpx::serialization::output_archive>(ar, n);                   \
  }                                                                           \
  HPX_SERIALIZATION_SPLIT_MEMBER();                                           \
  HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS(Class);                             \
/**/

#define HPX_SERIALIZATION_POLYMORPHIC_SPLITTED(Class)                         \
  virtual void load(hpx::serialization::input_archive& ar, unsigned n)        \
  {                                                                           \
    load<hpx::serialization::input_archive>(ar, n);                           \
  }                                                                           \
  virtual void save(hpx::serialization::output_archive& ar, unsigned n) const \
  {                                                                           \
    static bool register_class = (                                            \
      hpx::serialization::polymorphic_factory::instance().                    \
        register_class(                                                       \
          Class::hpx_serialization_get_hash(),                                \
          &Class::factory_function                                            \
        ),                                                                    \
      true                                                                    \
    );                                                                        \
    save<hpx::serialization::output_archive>(ar, n);                          \
  }                                                                           \
  HPX_SERIALIZATION_ADD_INTRUSIVE_MEMBERS(Class);                             \
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
  virtual boost::uint64_t hpx_serialization_get_hash() const = 0;             \
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
  virtual boost::uint64_t hpx_serialization_get_hash() const = 0;             \
/**/


#endif
