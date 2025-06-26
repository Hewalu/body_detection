import requests
import json

esp_ip = 'http://192.168.178.58:8080'

headers = {
    'Content-Type': 'application/json'
}


def send_data(data):
    try:
        response = requests.post(esp_ip, data=json.dumps(data), headers=headers, timeout=5)
        print(f"Antwort: {response.text}")
        print("-" * 50)

    except requests.exceptions.RequestException as e:
        print(f"Fehler beim Senden der Anfrage: {e}")
    except Exception as e:
        print(f"Unerwarteter Fehler: {e}")
