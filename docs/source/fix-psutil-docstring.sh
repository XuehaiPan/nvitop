#!/usr/bin/env bash

# shellcheck disable=SC2312
exec sed -i -E 's/^     process identity for every yielded instance$/  \0/' "$(python3 -c "print(__import__('psutil').__file__)")"
