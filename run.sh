#!/usr/bin/env bash
set -e

rm -f output_*.png

xvfb-run my_venv/bin/scenic --count 10 ./test.sc
