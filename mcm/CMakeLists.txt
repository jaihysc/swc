cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Initialise pico_sdk from installed location
# (note this can come from environment, CMake cache etc)

# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.2.0)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.2.0)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================

set(PICO_BOARD pico2_w CACHE STRING "Board type")

# Pull in Raspberry Pi Pico SDK (must be before project)
include(pico_sdk_import.cmake)
project(mcm C CXX ASM)

# Initialise the Raspberry Pi Pico SDK
pico_sdk_init()
add_executable(mcm
    main.c
    status.c
)

target_link_libraries(mcm
    pico_stdlib
    pico_btstack_ble
    pico_btstack_cyw43
    pico_cyw43_arch_none
    hardware_adc
)

target_include_directories(mcm PRIVATE
    ${CMAKE_CURRENT_LIST_DIR} # For btstack_config.h
)

target_compile_definitions(mcm PRIVATE
    RUNNING_AS_CLIENT=0
)

pico_btstack_make_gatt_header(mcm PRIVATE "${CMAKE_CURRENT_LIST_DIR}/profile.gatt")

# create map/bin/hex file etc.
pico_add_extra_outputs(mcm)
