import os
from .compare import main

print("\n--- GENERALIZATION TEST ---")
print("Testing policy on entirely distinct random seeded environments.")
print("The Overseer agent has NEVER seen these seeds or agent identites before.")

if __name__ == "__main__":
    main()
