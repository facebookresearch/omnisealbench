
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import atexit
import importlib
import os

from omnisealbench.commands.help import (help_eval, help_main, logo,
                                         print_audio_help, print_image_help,
                                         print_info_help,
                                         print_leaderboard_help,
                                         print_video_help)
from omnisealbench.commands.info import InfoCommand
from omnisealbench.commands.leaderboard import LeaderboardCommand
from omnisealbench.commands.utils import GracefulExit
from omnisealbench.utils.distributed import cleanup_distributed
from omnisealbench.utils.thirdparty import install_lib

atexit.register(cleanup_distributed)


def load_eval_commands(command_type: str, command_class: str):
    module_name = "omnisealbench.commands." + command_type
    module_ = importlib.import_module(module_name)
    if hasattr(module_, command_class):
        command_class_ = getattr(module_, command_class)
        return command_class_()


class EvalCommand:
    """Handle different evaluation commands."""

    def audio(self):
        """Audio commands."""
        return load_eval_commands("audio", "AudioCommand")

    def image(self):
        """Image commands."""
        return load_eval_commands("image", "ImageCommand")

    def video(self):
        """Video commands."""
        return load_eval_commands("video", "VideoCommand")

def main():
    import sys
    if "--help" in sys.argv:
        logo()
        if len(sys.argv) > 2:
            cmd = sys.argv[1]
            if cmd == "eval":
                command = sys.argv[2]
                if command == "audio":
                    print_audio_help()
                elif command == "image":
                    print_image_help()
                elif command == "video":
                    print_video_help()
                else:
                    help_eval()
            elif cmd == "leaderboard":
                print_leaderboard_help()
            elif cmd == "info":
                print_info_help()
            else:
                print(f"Unknown command: {cmd}")
                print("Use 'omniseal --help' for a list of available commands.")
                sys.exit(0)
        else:
            help_main()
        sys.exit(0)
    else:

        # Install Fire on-the-fly
        install_lib("fire")
        install_lib("rich")

        try:
            from fire import Fire

            Fire({"eval": EvalCommand, "leaderboard": LeaderboardCommand, "info": InfoCommand})

        # Handle error messages in cleaner way
        except (AssertionError, ImportError, ValueError, ModuleNotFoundError, FileNotFoundError, GracefulExit) as e:
            print(f"Error: {e}\n")
            if os.environ.get("OMNISEAL_DEBUG", "0") == "1":
                raise e
            else:
                print("If you need more information, please set the environment variable OMNISEAL_DEBUG=1 and rerun.")


if __name__ == "__main__":
    main()
