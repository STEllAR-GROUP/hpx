message ("source dir is " ${SCRIPT_SOURCE_DIR})
message ("dest dir is " ${SCRIPT_DEST_DIR})

configure_file(
  ${SCRIPT_SOURCE_DIR}/${SCRIPT_NAME}.sh.in 
  ${SCRIPT_DEST_DIR}/${SCRIPT_NAME}.sh
  @ONLY
)
