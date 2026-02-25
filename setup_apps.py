from AppOpener import open, give_appnames

print("--- INITIALIZING APP DATABASE ---")
print("I need to scan your computer to know what apps are installed.")
print("This might take 10-20 seconds.")

# This command forces a scan of your installed applications
give_appnames()

print("\n--- TEST RUN ---")
try:
    # Try opening Notepad to verify it works
    print("Attempting to open Notepad...")
    open("notepad", match_closest=True)
    print("SUCCESS! Apps are now indexed.")
except:
    print("Still having trouble. Ensure you are running as Administrator if required.")