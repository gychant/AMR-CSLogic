import requests


def request_wrapper(url, query, result_type="json"):
    try:
        print("query:", query)
        return requests.get(url, params={'format': result_type, 'query': query})
    except requests.exceptions.Timeout:
        print("TIMEOUT while processing \n" + query)
        return
    except requests.exceptions.TooManyRedirects:
        print("WRONG URL while processing \n" + query)
        return
    except requests.exceptions.RequestException as e:
        print("CATASTROPHIC ERROR " + str(e) + " while processing \n" + query)
        return
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print("Unknown error " + str(e) + " while processing  \n" + query)
        return

