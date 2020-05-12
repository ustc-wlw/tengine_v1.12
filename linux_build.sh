export PATH=/home/ffh/work/local/protobuf_3_0_0/bin:$PATH
export LD_LIBRARY_PATH=/home/ffh/work/local/protobuf_3_0_0/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/ffh/work/local/protobuf_3_0_0/bin:$LD_LIBRARY_PATH
export CPATH=/home/ffh/work/local/protobuf_3_0_0/include:$CPATH
# sudo cp /home/ffh/work/local/protobuf_3_0_0/lib/libprotobuf.so.10.0.0 /usr/lib/libprotobuf.so
# cmake -DPROTOBUF_DIR=/home/ffh/work/local/protobuf_3_0_0 \
#       -DCONFIG_ARCH_ARM64=1 \
#     ..
make -j32