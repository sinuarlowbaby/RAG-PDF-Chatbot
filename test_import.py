import sys
import traceback
sys.path.insert(0, './app')
try:
    import app.app
    print("\nSUCCESS")
except Exception as e:
    print("\nFAILED WITH ERROR:")
    traceback.print_exc()
