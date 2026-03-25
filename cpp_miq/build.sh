#!/usr/bin/env bash
# Build the run_miq executable.
#
# Usage (from the project root):
#   bash cpp_miq/build.sh
#
# Or from inside cpp_miq/:
#   bash build.sh
#
# The binary ends up at  cpp_miq/build/run_miq.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

echo "=== NPAQ / run_miq build ==="
echo "Source : $SCRIPT_DIR"
echo "Build  : $BUILD_DIR"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure (FetchContent will download libigl + CoMISo on first run).
# To enable libQEx robust quad extraction (requires OpenMesh):
#   sudo apt-get install libopenmesh-dev
#   bash build.sh --libqex
LIBQEX_FLAG=OFF
for arg in "$@"; do [ "$arg" = "--libqex" ] && LIBQEX_FLAG=ON; done

cmake "$SCRIPT_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBIGL_COPYLEFT_COMISO=ON \
    -DLIBIGL_WITH_OPENGL=OFF \
    -DLIBIGL_WITH_OPENGL_GLFW=OFF \
    -DLIBIGL_WITH_VIEWER=OFF \
    -DNPAQ_USE_LIBQEX="$LIBQEX_FLAG"

# Build
cmake --build . --config Release -- -j"$(nproc 2>/dev/null || echo 4)"

echo ""
echo "=== Build complete ==="
echo "Binary: $BUILD_DIR/run_miq"
echo ""
echo "Quick test:"
echo "  $BUILD_DIR/run_miq --help"
