find_package(Boost 1.48.0 REQUIRED COMPONENTS system filesystem regex serialization)
find_package(OpenMP REQUIRED)
find_library(ATLAS_LIBRARY atlas
			HINTS ${ATLAS_ROOT}/lib ${CBLAS_ROOT}/lib /opt/local/lib /usr/lib64/atlas/
        )
message(STATUS "ATLAS location is: " ${ATLAS_LIBRARY})
find_library(CBLAS_LIBRARY cblas
			HINTS ${ATLAS_ROOT}/lib ${CBLAS_ROOT}/lib /opt/local/lib /usr/lib64/atlas/
		)
message(STATUS "CBLAS location is: " ${CBLAS_LIBRARY})
if(Boost_FOUND)
    message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")
    message(STATUS "Boost_VERSION: ${Boost_VERSION}")
    include_directories(${BOOST_ROOT})
    include_directories(${Boost_INCLUDE_DIRS})
    link_directories(${Boost_LIBRARY_DIRS})
    add_executable(svgpm main3.cpp) #Main file
    target_link_libraries(${PROJECT_NAME} PRIVATE svgpm-libs ${Boost_LIBRARIES} OpenMP::OpenMP_CXX ${ATLAS_LIBRARY} ${CBLAS_LIBRARY}) 
endif()




# target_link_libraries(svgpm PRIVATE svgpm-libs)
