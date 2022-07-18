# This file is part of nvitop, the interactive NVIDIA-GPU process viewer.
# License: GNU GPL version 3.

# pylint: disable=missing-module-docstring


def add_scalar_dict(writer, main_tag, tag_scalar_dict, global_step=None, walltime=None):
    """Batched version of `writer.add_scalar`"""
    for tag, scalar in tag_scalar_dict.items():
        writer.add_scalar(
            '{}/{}'.format(main_tag, tag), scalar, global_step=global_step, walltime=walltime
        )
