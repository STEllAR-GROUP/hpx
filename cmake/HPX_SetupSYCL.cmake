if(HPX_WITH_SYCL)
  add_compile_options(-fno-sycl)
  hpx_add_config_define(HPX_HAVE_SYCL)
  # TODO do we really have compute? What does that define implicate?
  hpx_add_config_define(HPX_HAVE_COMPUTE)
endif()
