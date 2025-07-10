import socket
import json
from typing import Any
import threading


class DataSender:
    """
    Sammelt Daten und sendet sie als JSON-Objekt per UDP-Broadcast an alle ESPs im Netzwerk.

    Beispiel:
        sender = DataSender("192.168.161.255", 8080)
        sender.add_data("left_arm_angle", 90)
        sender.add_data("right_arm_angle", 150)
        sender.send()
    """

    def __init__(self, broadcast_ip: str, port: int):
        self.broadcast_address = (broadcast_ip, port)
        self._data = {}
        self._lock = threading.Lock()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    def add_data(self, key: str, value: Any):
        """Fügt dem Datenobjekt ein Schlüssel-Wert-Paar hinzu."""
        with self._lock:
            self._data[key] = value

    def send(self):
        """Sendet das gesammelte Datenobjekt per UDP-Broadcast."""
        with self._lock:
            if not self._data:
                return

            # Wenn mode_switch vorhanden ist, nur diesen Key senden
            if "mode_switch" in self._data:
                data_to_send = {"mode_switch": self._data["mode_switch"]}
            else:
                data_to_send = self._data.copy()

            self._data.clear()

        try:
            message = json.dumps(data_to_send).encode('utf-8')
            self.socket.sendto(message, self.broadcast_address)
            print(f"Gesendet an {self.broadcast_address[0]}:{self.broadcast_address[1]} -> {json.dumps(data_to_send)}")
        except Exception as e:
            print(f"Fehler beim Senden: {e}")

        print("-" * 50)
