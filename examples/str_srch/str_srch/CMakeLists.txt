set(dependencies
    distributing_factory_component
    dataflow_component)

add_hpx_component(str_srch
  DEPENDENCIES ${dependencies}
  FOLDER "Examples/Str_srch"
  AUTOGLOB)
