# Pinned by digest so the fuzzing build is reproducible. OSS-Fuzz base images move
# quickly; the docker entry in .github/dependabot.yml keeps this digest current.
FROM gcr.io/oss-fuzz-base/base-builder-python@sha256:58f80c90a90944b68b8e70b98e38997ee9396225311cfb0f3aea78671446ab4e

COPY . $SRC/astra-synthetic-gating-demo
WORKDIR $SRC/astra-synthetic-gating-demo
COPY ./.clusterfuzzlite/build.sh $SRC/
