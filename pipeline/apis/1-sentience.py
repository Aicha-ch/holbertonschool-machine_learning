#!/usr/bin/env python3
"""
Retrieve the home planets of all sentient species.
"""

import requests


def sentientPlanets():
    """
    Retrieve the home planets of all sentient species.
    """
    base_url = "https://swapi-api.hbtn.io/api/species/"
    planets = set()
    sentient_types = {"sentient"}
    while base_url:
        response = requests.get(base_url)
        if response.status_code != 200:
            break
        data = response.json()
        for species in data.get("results", []):
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()
            if (classification in sentient_types or
                    designation in sentient_types):
                homeworld = species.get("homeworld")
                if homeworld:
                    planet_response = requests.get(homeworld)
                    if planet_response.status_code == 200:
                        planet_data = planet_response.json()
                        planets.add(planet_data.get("name", "unknown"))
                    else:
                        planets.add("unknown")
        base_url = data.get("next")
    return sorted(planets)
