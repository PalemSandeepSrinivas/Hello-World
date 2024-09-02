import keys
from dotenv import load_dotenv
import os

def configure():
    load_dotenv()

def main():
    configure()
    print(keys.some_key)
    print(os.getenv('api_key'))


main()