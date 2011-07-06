# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

source_group("Source Files\\Server" FILES point_geometry/point_geometry.cpp)
source_group("Source Files" FILES point_geometry_client.cpp)

add_hpx_component(point_geometry
                  MODULE point_geometry
                  SOURCES point_geometry/point_geometry.cpp)

add_hpx_executable(point_geometry_client
                    MODULE queue
                    SOURCES point_geometry_client.cpp
                    DEPENDENCIES point_geometry_component)

add_hpx_pseudo_target(examples.geometry.point_geometry)

add_hpx_pseudo_dependencies(examples.geometry examples.geometry.point_geometry)

add_hpx_pseudo_dependencies(examples.geometry.point_geometry
                            point_geometry_client_exe) 

