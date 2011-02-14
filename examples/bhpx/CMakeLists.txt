# FIXME: this file is intentionally lacking a copyright notice as I do not know
# the authors preference.

# just recurse into all components subdirectories and execute the scripts there
set(hpx_BHPX_SUBDIRS 
    tlf
    itn)

foreach(hpx_BHPX_SUBDIR ${hpx_ACCUMULATOR_SUBDIRS})
  add_subdirectory(${hpx_BHPX_SUBDIR})
endforeach()

# define build target
set(bhc_SOURCES 
    bhc.cpp)

# add example executable
add_hpx_executable(bhc
  SOURCES ${bhc_SOURCES}
  DEPENDENCIES tlf_component 
               itn_component
               distributing_factory_component)

# add a dependency to the pseudo-target  
add_dependencies(examples.bhpx bhc) 

