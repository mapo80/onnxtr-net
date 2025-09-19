# Repository Guidelines

## .NET Installation

To install the .NET SDK inside this container:

1. Ensure the installation helper is executable: `chmod +x ./donetinstall`.
2. Run the helper script: `./donetinstall -Channel 8.0`.
3. Export the toolchain to the current session: `export DOTNET_ROOT="$HOME/.dotnet"` and `export PATH="$DOTNET_ROOT:$PATH"`.

Always re-run step 3 in new shells before invoking `dotnet`.
