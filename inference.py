"""Root benchmark entrypoint for MEVerse inference."""

import asyncio

from meverse.baseline_runner import main


if __name__ == "__main__":
    asyncio.run(main())
