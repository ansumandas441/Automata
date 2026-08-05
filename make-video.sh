#!/bin/sh
# Two recordings in, one edited video out.
#
#   ./make-video.sh screen.mp4 camera.mp4 [out.mp4]
#
# Runs init -> analyze -> plan -> render, stopping at the first failure. Working
# files are written next to the output so you can re-plan or hand-edit later
# without re-running the slow half.
set -eu

if [ $# -lt 2 ]; then
  echo "usage: $0 <screen.mp4> <camera.mp4> [out.mp4]" >&2
  echo "  --talking-head as a 4th argument if your camera sits on the screen" >&2
  exit 2
fi

screen=$1
camera=$2
out=${3:-out.mp4}
extra=${4:-}

here=$(cd "$(dirname "$0")" && pwd)
automata="$here/.venv/bin/automata"
[ -x "$automata" ] || automata=automata

work=$(dirname "$out")/$(basename "$out" .mp4).project.json

echo "==> measuring"
"$automata" init "$screen" "$camera" -o "$work" --force $extra

echo
echo "==> analysing (the slow part; runs once)"
"$automata" analyze "$work" --force

echo
echo "==> deciding the edit"
"$automata" plan "$work"

echo
echo "==> rendering"
"$automata" render "$work" -o "$out"

echo
echo "Done: $out"
echo
echo "To change the edit without re-analysing:"
echo "  edit the \"director\" block in $work, then:"
echo "  $automata plan $work && $automata render $work -o $out"
