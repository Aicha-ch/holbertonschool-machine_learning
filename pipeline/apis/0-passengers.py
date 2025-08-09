#!/usr/bin/env python3
"""
Retrieve ships.
"""

import requests


def availableShips(passengerCount):
    """
    Retrieve ships.
    """
    base_url = "https://swapi-api.hbtn.io/api/starships/"
    ships = []

    while base_url:
        response = requests.get(base_url)
        if response.status_code != 200:
            break

        data = response.json()
        for ship in data.get("results", []):
            passengers = ship.get("passengers", "0").replace(",", "")
            if passengers.isdigit() and int(passengers) >= passengerCount:
                ships.append(ship.get("name"))
        base_url = data.get("next")
    return ships