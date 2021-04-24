
add_library(
    svgpm-libs
    puppy/Context.hpp
    puppy/Primitive.hpp
    puppy/PrimitiveHandle.hpp
    puppy/PrimitiveInline.hpp
    puppy/Puppy.hpp
    puppy/Randomizer.hpp
    puppy/TokenT.hpp
    puppy/Tree.hpp
    DecisionTree.h
    DecisionTree.cpp
    GPAProgram.h
    GPAProgram.cpp
    GPAttributePrimitive.h
    GPAttributePrimitive.cpp
    GPSVMGM.h
    GPSVMGM.cpp
    Primitive.cpp
    Puppy.cpp
    Tree.cpp
    BinaryConfusionMatrix.h
)

# target_include_directories(svgpm-libs PUBLIC "/usr/local/include/")
target_include_directories(svgpm-libs PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" )
target_link_libraries(svgpm-libs PUBLIC "/usr/local/lib/libshark.a")