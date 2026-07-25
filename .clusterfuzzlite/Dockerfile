# Pinned by digest so the fuzzing build is reproducible. OSS-Fuzz base images move
# quickly; the docker entry in .github/dependabot.yml keeps this digest current.
FROM gcr.io/oss-fuzz-base/base-builder-python@sha256:6a6927b54c4cfd5de1bf4aa638d8db5d94a3c442fa9282b6cf864396c5ba23d1

COPY . $SRC/astra-synthetic-gating-demo
WORKDIR $SRC/astra-synthetic-gating-demo
COPY ./.clusterfuzzlite/build.sh $SRC/
