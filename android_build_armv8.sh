#!/bin/bash
   #  -DANDROID_ARM_NEON=ON \
   #  -DANDROID_STL=gnustl_shared \
export PATH=/data_1/Projects/cjh_tengine/x86/protobuf_3_0_0/bin:$PATH
export LD_LIBRARY_PATH=/data_1/Projects/cjh_tengine/x86/protobuf_3_0_0/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data_1/Projects/cjh_tengine/x86/protobuf_3_0_0/bin:$LD_LIBRARY_PATH
export CPATH=/data_1/Projects/cjh_tengine/x86/protobuf_3_0_0/include:$CPATH

PROTOBUF_PATH=/data_1/Projects/android/android_gnu_shared/protobuf_lib
BLAS_PATH=/data_1/Projects/android/android_gnu_shared/openbla020_android
ARCH_TYPE=ARMv8
ANDROID_NDK=/data_1/Projects/android/android-ndk-r16

#cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
#    -DANDROID_ABI="arm64-v8a" \
#    -DCONFIG_ARCH_ARM64=ON \
#    -DANDROID_PLATFORM=android-22 \
#    -DANDROID_STL=c++_shared \
#    -DPROTOBUF_DIR=$PROTOBUF_PATH \
#    -DCONFIG_ARCH_BLAS=ON \
#    -DANDROID_ARM_NEON=ON \
#    -DANDROID_ALLOW_UNDEFINED_SYMBOLS=TRUE
#    ..

cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_PLATFORM=android-22 \
    -DANDROID_STL=c++_shared \
    -DANDROID_ARM_NEON=ON \
    -DCONFIG_ARCH_ARM64=ON \
    -DANDROID_ALLOW_UNDEFINED_SYMBOLS=TRUE \
    ..

make -j8 && make install
