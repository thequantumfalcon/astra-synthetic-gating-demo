# Pinned by digest so the fuzzing build is reproducible. OSS-Fuzz base images move
# quickly; the docker entry in .github/dependabot.yml keeps this digest current.
FROM gcr.io/oss-fuzz-base/base-builder-python@sha256:6b2b3a7e4a2da50de47f94925cffbe34747e1aae4f33d7f07ba6c681dd648b23

COPY . $SRC/astra-synthetic-gating-demo
WORKDIR $SRC/astra-synthetic-gating-demo
COPY ./.clusterfuzzlite/build.sh $SRC/
