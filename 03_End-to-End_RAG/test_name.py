print(f"__name__ is: {__name__}")

def main():
    print("main() function is running!")

if __name__ == "__main__":
    print("This script is being run directly")
    main()
else:
    print("This script is being imported")
