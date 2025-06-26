import requests
import json
from typing import Any
import threading
import time


class DataSender:
    """
    Sammelt Daten und sendet sie als JSON-Objekt an einen ESP.

    Beispiel fuer die Verwendung:
    sender = DataSender()
    sender.add_data("armsMovement", "up")
    sender.add_data("armsAngle", 90)
    sender.send()
    """

    def __init__(self, esp_ip='http://192.168.178.58:8080'):
        self.esp_ip = esp_ip
        self.headers = {'Content-Type': 'application/json'}
        self._data = {}
        self._lock = threading.Lock()
        self._connection_lost = False
        self._last_error_print_time = 0
        self._error_print_interval = 10  # In Sekunden

    def add_data(self, key: str, value: Any):
        """Fuegt dem Datenobjekt ein Schluessel-Wert-Paar hinzu."""
        with self._lock:
            self._data[key] = value

    def send(self):
        """
        Sendet das gesammelte Datenobjekt thread-sicher.
        Die Daten werden nach dem Kopieren zum Senden zurueckgesetzt.
        """
        data_to_send = None
        with self._lock:
            if self._data:
                data_to_send = self._data.copy()
                self._data.clear()

        if not data_to_send:
            return

        try:
            response = requests.post(self.esp_ip, data=json.dumps(data_to_send), headers=self.headers, timeout=2)

            if self._connection_lost:
                print("Verbindung zum ESP wiederhergestellt.")
                self._connection_lost = False

            print(f"Daten gesendet: {json.dumps(data_to_send)}")
            print(f"Antwort: {response.text}")
            print("-" * 50)
        except requests.exceptions.RequestException:
            current_time = time.time()
            if not self._connection_lost or (current_time - self._last_error_print_time) > self._error_print_interval:
                print("Verbindung zum ESP fehlgeschlagen. ueberpruefen Sie, ob das Geraet aktiv ist.")
                self._last_error_print_time = current_time
            self._connection_lost = True
        except Exception as e:
            print(f"Unerwarteter Fehler: {e}")
