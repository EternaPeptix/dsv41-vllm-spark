#!/bin/bash
# r2-skinny: run ON spark-eight (brief rule 4: spark-eight only, nice 19, MAX_JOBS=1, venv toolchain).
# The Mac-side driver checks spark-one's CLUSTER_LOCK before starting and every 10 s during the build, and kills
# this build (pkill -f r2-skinny/kernel) if a lock appears.
set -u
D=$HOME/climb-scratch/r2-skinny/kernel
cd "$D" || exit 2
rm -rf build
echo "build start $(date -u +%FT%TZ) host $(hostname)" > build.log
PATH=$HOME/venvs/vllm032/bin:$PATH MAX_JOBS=1 nice -n 19 $HOME/venvs/vllm032/bin/python $D/build.py >> build.log 2>&1
echo "BUILD_RC=$?" >> build.log
echo "build end $(date -u +%FT%TZ)" >> build.log
sha256sum $D/build/*.so >> build.log 2>&1
