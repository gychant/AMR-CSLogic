import argparse
import json
from amr_verbnet_semantics.service.amr import LocalAMRClient

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', '-i', action='store_true',
                        help="Whether to run the interactive mode")
    parser.add_argument('--output_json', '-j', action='store_true',
                        help="Whether to output as JSON string")

    args = parser.parse_args()

    amr_client = LocalAMRClient()
    
    if args.interactive:
        while True:
            try:
                text = input(">> Input text to parse: \n")
                print()
                amr = amr_client.get_amr(text)
                print("\n\n")
                print(amr)
                print()
                print(text)
                print()
                if args.output_json:
                    data = [{
                        "sent": text,
                        "amr": amr
                    }]
                    print(json.dumps(data))
            except KeyboardInterrupt:
                raise 

    list_text = [
        "I loved him writing novels.",
        "I admired him for his honesty."
    ]
    for text in list_text:
        amr = amr_client.get_amr(text)
        print(amr)

