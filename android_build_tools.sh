#!/bin/bash
   #  -DANDROID_ARM_NEON=ON \
   #  -DANDROID_STL=gnustl_shared \
export PATH=/data_1/Projects/cjh_tengine/x86/protobuf_3_0_0/bin:$PATH
export LD_LIBRARY_PATH=/data_1/Projects/cjh_tengine/x86/protobuf_3_0_0/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data_1/Projects/cjh_tengine/x86/protobuf_3_0_0/bin:$LD_LIBRARY_PATH
export CPATH=/data_1/Projects/cjh_tengine/x86/protobuf_3_0_0/include:$CPATH


cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/x86_convert_tool.gcc.toolchain.cmake ..

make -j8 && make install
