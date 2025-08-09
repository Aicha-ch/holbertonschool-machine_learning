#!/usr/bin/env python3
"""
Check API.
"""
import requests


def availableShips(passengerCount):
    """
    Returns a list of ship names.
    """
    base_url = "https://swapi.dev/api/starships/"
    ships = []

    while base_url:
        response = requests.get(base_url)
        if response.status_code != 200:
            break

        data = response.json()
        for ship in data.get('results', []):
            try:
                passengers = ship.get('passengers', '0').replace(',', '')
                if passengers.isdigit() and int(passengers) >= passengerCount:
                    ships.append(ship.get('name'))
            except ValueError:
                continue

        base_url = data.get('next')

    return ships