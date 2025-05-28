#!/bin/bash

shiv -e nvitop_exporter.__main__:main -o nvitop-exporter --site-packages . nvitop prometheus-client
