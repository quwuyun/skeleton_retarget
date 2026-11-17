"""Small command router for the independent retargeting workspace.

Run from the project root with the uv-managed environment::

    uv run python -m skeleton_retarget --help
    uv run python -m skeleton_retarget skeleton17 retarget --help
"""

import argparse
import runpy
import sys

WORKFLOWS = {
    "skeleton17": {
        "prepare": "prepare_skeleton17",
        "retarget": "bvh_to_skeleton17",
        "validate": "validate_skeleton17",
        "batch": "bvh_to_skeleton17_dataset",
        "behaviors": "build_behavior_set",
        "mirror": "mirror_skeleton17",
    },
    "skeleton31": {
        "prepare": "prepare_skeleton31",
        "retarget": "bvh_to_skeleton31",
        "validate": "validate_skeleton31",
        "batch": "bvh_to_skeleton31_dataset",
        "mimickit": "mimickit_to_skeleton31",
    },
}
TOOLS = {
    "play": "vis_motion",
    "preview": "preview_motion",
    "select": "select_walk_cycles",
    "normalize-heading": "normalize_heading",
}


def main():
    parser = argparse.ArgumentParser(
        description="Motion retargeting commands for the skeleton17 and skeleton31 models.",
        epilog="Append --help to any command to see its options.",
    )
    parser.add_argument("command", choices=[*WORKFLOWS, *TOOLS], nargs="?")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        print("\nModel workflows:")
        for profile, commands in WORKFLOWS.items():
            print(f"  {profile}: {', '.join(commands)}")
        return
    if args.command in TOOLS:
        module = f"skeleton_retarget.scripts.{TOOLS[args.command]}"
        forwarded = args.arguments
    else:
        commands = WORKFLOWS[args.command]
        if not args.arguments or args.arguments[0] in {"-h", "--help"}:
            print(f"{args.command} commands: {', '.join(commands)}")
            print(f"Usage: python -m skeleton_retarget {args.command} COMMAND --help")
            return
        command, *forwarded = args.arguments
        if command not in commands:
            parser.error(f"{args.command}: choose one of {', '.join(commands)}")
        module = f"skeleton_retarget.scripts.{commands[command]}"
    sys.argv = [module, *forwarded]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
