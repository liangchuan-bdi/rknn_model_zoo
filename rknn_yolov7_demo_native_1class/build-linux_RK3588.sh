set -e

TARGET_SOC="rk3588"
#GCC_COMPILER=aarch64-linux-gnu
GCC_COMPILER=aarch64-rockchip-linux-gnu

export TOOL_CHAIN=$HOME/Development/rk3588/toolchain/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu

echo "TOOL_CHAIN: ${TOOL_CHAIN}"

export LD_LIBRARY_PATH=${TOOL_CHAIN}/lib64:$LD_LIBRARY_PATH
export CC=${TOOL_CHAIN}/bin/${GCC_COMPILER}-gcc
export CXX=${TOOL_CHAIN}/bin/${GCC_COMPILER}-g++

ROOT_PWD=$( cd "$( dirname $0 )" && cd -P "$( dirname "$SOURCE" )" && pwd )

# build
BUILD_DIR=${ROOT_PWD}/build/build_linux_aarch64

if [[ ! -d "${BUILD_DIR}" ]]; then
  mkdir -p ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake ../.. -DCMAKE_SYSTEM_NAME=Linux -DTARGET_SOC=${TARGET_SOC} -DCMAKE_BUILD_TYPE=Release
make -j4
make install
cd -
