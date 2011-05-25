// Copyright (c) 2010-2011 Dylan Stark
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying 
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PXGL_PXGL_GRAPHS_EDGE_TUPLE_20100917T1324)
#define PXGL_PXGL_GRAPHS_EDGE_TUPLE_20100917T1324

////////////////////////////////////////////////////////////////////////////////
namespace pxgl { namespace graphs { namespace server {
  struct edge_tuple
  {
    typedef unsigned long size_type;
  
    edge_tuple() 
    {}
  
    edge_tuple(size_type const & source, 
               size_type const & target, 
               size_type const & weight)
      : source_(source),
        target_(target),
        weight_(weight)
    {}
  
    size_type source(void) const { return source_; }
    size_type target(void) const { return target_; }
    size_type weight(void) const { return weight_; }
  
    private:
      // Serialization support
      friend class boost::serialization::access;
  
      template <typename Archive>
      void serialize(Archive& ar, const unsigned int)
      {
        ar & source_ & target_ & weight_;
      }
  
      // Data members
      size_type source_;
      size_type target_;
      size_type weight_;
  };
  typedef edge_tuple edge_tuple_type;
  typedef std::vector<edge_tuple> edge_tuples_type;
}}}

#endif
