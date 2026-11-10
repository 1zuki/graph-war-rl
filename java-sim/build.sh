#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="$root_dir/build"
classes_dir="$build_dir/classes"

mkdir -p "$classes_dir"
javac \
  -encoding UTF-8 \
  -d "$classes_dir" \
  "$root_dir/src/main/java/graphwarrl/HeadlessSimulator.java"
jar \
  --create \
  --file "$build_dir/graphwar-headless.jar" \
  --main-class graphwarrl.HeadlessSimulator \
  -C "$classes_dir" .
