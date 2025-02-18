#!/bin/bash
# This script finds the most recent core dump in the current directory,
# determines the executable linked to it, and then uses GDB to generate a full backtrace log.

# Find the most recent core dump file (assumes files start with "core")
CORE_FILE=$(ls -1 core* 2>/dev/null | head -n 1)
if [ -z "$CORE_FILE" ]; then
    echo "No core dump file found in the current directory."
    exit 1
fi

# Extract the associated executable from the core dump metadata using the 'file' command.
# The output is expected to contain a line like: "core: ELF 64-bit LSB core file ... from 'executable_name'"
EXE=$(file "$CORE_FILE" | grep -oP "from '.*?'" | sed "s/from '//;s/'//")
if [ -z "$EXE" ]; then
    echo "Could not determine the executable from the core dump metadata."
    exit 1
fi

echo "Core dump file: $CORE_FILE"
echo "Associated executable: $EXE"

# Use GDB to get a full backtrace and save it to a log file.
# The '-batch' option runs GDB non-interactively.
gdb -batch -ex "bt full" -ex "quit" "$EXE" "$CORE_FILE" > core_backtrace.log

if [ $? -eq 0 ]; then
    echo "Backtrace log successfully saved to 'core_backtrace.log'."
else
    echo "Failed to generate backtrace log."
fi

